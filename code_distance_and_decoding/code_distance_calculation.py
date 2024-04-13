from typing import List, Dict, Optional
import os
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix
import pickle
from tqdm import tqdm
import multiprocessing

from mip import Model, xsum, minimize, BINARY, OptimizationStatus

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
sys.path.append(parent_dir)

# Import the get_net_encoding_rate function from the helper_functions.py file
from helper_functions import (
    save_as_pickle,
    load_from_pickle,
    split_list,
)

### Surpress the print output of the css_code.test() ###
original_stdout = sys.stdout

# Setup logging
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


class CodeDistanceCalculator:
    """
    A class to calculate the code distance for LDPC codes. It handles scanning directories for new code files,
    calculating code distances, and saving the results. The calculations can include excluding specific directories,
    and managing intermediate results.

    Attributes:
        found_code_configs_dir (str): Directory where the found code configurations are stored.
        exclude_dirs (Optional[List[str]]): Directories to exclude from the code distance calculation.
        decoded_codes_dir (str): Directory to save decoded codes during the ongoing search.
        known_files (set): Set to keep track of files already processed.
    """

    def __init__(self, found_codes_dir: str, exclude_dirs: Optional[List[str]] = None, decoded_codes_dir: str = "intermediate_results_code_distance_during_ongoing_code_search"):
        """
        Initializes the CodeDistanceCalculator with specified directories and exclusion settings.

        Parameters:
            found_codes_dir (str): The directory where found code configurations are stored.
            exclude_dirs (Optional[List[str]]): Optional list of directories to exclude from scanning.
            decoded_codes_dir (str): The directory where decoded codes should be saved.
        """
        self.found_code_configs_dir = found_codes_dir
        self.exclude_dirs = exclude_dirs
        self.decoded_codes_dir = decoded_codes_dir
        self.known_files = set()

    def check_for_new_files(self) -> List[str]:
        """
        Scans the specified directory and its subdirectories for new files not listed in `known_files`,
        excluding the directories specified in `exclude_dirs`.

        Returns:
            List[str]: A list of new file paths found in the directory and its subdirectories,
                       excluding the specified directories.
        """
        if self.exclude_dirs is None:
            self.exclude_dirs = set()
        new_files = []
        for root, dirs, files in os.walk(self.found_code_configs_dir):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]  # Modify dirs in-place to exclude certain directories
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path) and file_name not in self.known_files:
                    new_files.append(file_path)
        return new_files

    def start_calculation(self) -> None:
        """
        Starts the process of checking for new files continuously and calculating code distances
        for new configurations. If new files are found, they are processed and added to known_files.
        The function waits for 30 minutes before checking again.
        """

        while True:
            new_files = self.check_for_new_files()
            if new_files:
                for file_path in new_files:
                    if file_path.endswith(".pickle"):
                        original_subfolder_name = os.path.relpath(
                            os.path.dirname(file_path), self.found_code_configs_dir
                        )
                        logging.warning(
                            f"Processing file from subfolder: {original_subfolder_name}"
                        )
                        code_configs = load_from_pickle(file_path)
                        self.get_code_distance_parallel(
                            code_configs, original_subfolder_name
                        )
                        self.known_files.add(
                            os.path.relpath(file_path, self.found_code_configs_dir)
                        )  # Update known_files with the new file
                    else:
                        continue
            else:
                logging.warning(
                    "No new files found. Waiting for 30 minutes before checking again."
                )
            # Wait for 30 minutes before next check
            time.sleep(1800)


    def load_and_unify_intermediate_results(self, subfolder: str) -> None:
        """
        Loads intermediate results from the specified subfolder, unifies them into a single file,
        and cleans up the intermediate files.

        Parameters:
            subfolder (str): The directory containing the intermediate result files to unify.
        """
        unified_configs, files_to_delete = [], [] # List to keep track of files to delete after unification

        # Count existing unified pickle files to determine the index for the new file
        existing_unified_files = [
            f for f in os.listdir(subfolder) if f.startswith("unified_codes_with_distance_")
        ]
        next_file_index = len(existing_unified_files) + 1

        for filename in os.listdir(subfolder):
            if filename.startswith("codes_with_distance") and filename.endswith(".pickle"):
                file_path = os.path.join(subfolder, filename)
                # Add this file to the list of files to delete
                files_to_delete.append(file_path)

                # Load the content of each pickle file and extend the unified list
                configs = load_from_pickle(file_path)
                unified_configs.extend(configs)

        # Save the unified results
        unified_file_path = os.path.join(
            subfolder, f"unified_codes_with_distance_{next_file_index}.pickle"
        )
        save_as_pickle(unified_file_path, unified_configs)
        logging.warning("Saved unified results to: {}".format(unified_file_path))

        # Delete the intermediate files
        for file_path in files_to_delete:
            os.remove(file_path)


    def calculate_code_distance(self, code_config: Dict) -> Dict:
        """
        Calculates the code distance for a quantum error correction (QEC) code configuration and updates the configuration dictionary.

        Parameters:
            code_config (Dict): The configuration dictionary of a QEC code.

        Returns:
            Dict: The updated configuration dictionary including the calculated code distance.
        """

        try:
            # computes the minimum Hamming weight of a binary vector x such that
            # stab @ x = 0 mod 2
            # logicOp @ x = 1 mod 2
            # here stab is a binary matrix and logicOp is a binary vector
            def distance_test(stab, logicOp, code_config) -> int:
                # number of qubits
                n = stab.shape[1]
                # number of stabilizers
                m = stab.shape[0]

                # maximum stabilizer weight
                wstab = np.max([np.sum(stab[i, :]) for i in range(m)])
                # weight of the logical operator
                wlog = np.count_nonzero(logicOp)
                # how many slack variables are needed to express orthogonality constraints modulo two
                num_anc_stab = int(np.ceil(np.log2(wstab)))
                num_anc_logical = int(np.ceil(np.log2(wlog)))
                # total number of variables
                num_var = n + m * num_anc_stab + num_anc_logical

                model = Model()
                model.verbose = 0
                x = [model.add_var(var_type=BINARY) for i in range(num_var)]
                model.objective = minimize(xsum(x[i] for i in range(n)))

                # orthogonality to rows of stab constraints
                for row in range(m):
                    weight = [0] * num_var
                    supp = np.nonzero(stab[row, :])[0]
                    for q in supp:
                        weight[q] = 1
                    cnt = 1
                    for q in range(num_anc_stab):
                        weight[n + row * num_anc_stab + q] = -(1 << cnt)
                        cnt += 1
                    model += xsum(weight[i] * x[i] for i in range(num_var)) == 0

                # odd overlap with logicOp constraint
                supp = np.nonzero(logicOp)[0]
                weight = [0] * num_var
                for q in supp:
                    weight[q] = 1
                cnt = 1
                for q in range(num_anc_logical):
                    weight[n + m * num_anc_stab + q] = -(1 << cnt)
                    cnt += 1
                model += xsum(weight[i] * x[i] for i in range(num_var)) == 1

                model.optimize()
                if (
                    model.status != OptimizationStatus.OPTIMAL
                ):  # If not optimal, record the status and raise an exception
                    code_config["distance"] = f"Non-optimal solution: {model.status}"
                    raise Exception("Non-optimal solution: {}".format(model.status))

                opt_val = sum([x[i].x for i in range(n)]) # type: ignore

                return int(opt_val)

            distance_list = []
            hx = code_config["hx"].toarray() if isinstance(code_config["hx"], csr_matrix) else code_config["hx"]
            lx = code_config["lx"].toarray() if isinstance(code_config["lx"], csr_matrix) else code_config["lx"]
            
            for i in range(code_config["num_log_qubits"]):
                ## add stabilizer to logical. This seems to fix some issues with distance test
                ## in paritcular, infeasability error and wrong distances seems to be fixed with this
                ## it seems the code works better if the logical has higher weight than stabilizer
                ## or more slack ancillas for logical are added. This has no impact on problem, yet seems to fix it
                ## note that one can always add stabilizer to logical
                w = distance_test(hx, lx[i, :]+hx[0, :], code_config)
                distance_list.append(w)
            code_distance = min(distance_list)            

            code_config["distance_summary"] = {
                "distance": code_distance,
                "distance_list": distance_list,
            }
            return code_config

        except Exception as e:
            if "Non-optimal solution" not in str(e):
                logging.warning(
                    "An error happened in the distance calculation: {}".format(e)
                )
                # Indicate an error occurred
                code_config["distance_summary"] = {
                    "distance": f"Error in code distance calculation: {e}",
                    "distance_list": distance_list,
                }
                    
            return code_config


    def get_code_distance_parallel(
        self, code_configs: List[Dict], original_subfolder_name: str
    ) -> None:
        """
        Handles the parallel execution of code distance calculations for a list of code configurations.

        Parameters:
            code_configs (List[Dict]): List of code configurations to calculate distance for.
            original_subfolder_name (str): Name of the subfolder where the results should be saved.

        """
        # The folder where results will be saved, named after the original subfolder
        result_subfolder = os.path.join(
            self.decoded_codes_dir, 
            original_subfolder_name,
        )

        # Determine the number of processes to use
        num_processes = multiprocessing.cpu_count()
        num_chunks = len(code_configs) // 200
        if num_chunks < 1:
            num_chunks = 1
        chunked_list = split_list(code_configs, num_chunks)  # Split the list into chunks

        start_time = time.time()
        logging.warning(
            "------------------ START CODE DISTANCE CALCULATION ------------------"
        )
        logging.warning(f"Number of processes: {num_processes}")
        logging.warning(f"Number of code configurations: {len(code_configs)}")

        for i, chunk in enumerate(tqdm(chunked_list)):
            logging.warning(f"Number of codes in Chunk {i+1}: {len(chunk)}")

            with multiprocessing.Pool(processes=num_processes) as pool:
                code_configs_with_distance = pool.map(self.calculate_code_distance, chunk)

            # Save intermediate results in the new subfolder
            file_path = os.path.join(
                result_subfolder, f"codes_with_distance_{i+1}.pickle"
            )
            save_as_pickle(file_path, code_configs_with_distance)

        logging.warning(
            "------------------ FINISHED CODE DISTANCE CALCULATION ------------------"
        )
        logging.warning(
            f"Distance calculation took: {round((time.time() - start_time) / 3600.0, 2)} hours."
        )

        # Once all chunks are processed, unify the intermediate results and clean up
        try:
            self.load_and_unify_intermediate_results(subfolder=result_subfolder)
        except Exception as e:
            logging.warning(
                f"An error occurred while unifying the intermediate results: {e}"
            )
            return
