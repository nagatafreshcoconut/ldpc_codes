from typing import List, Dict, Optional
import os
import sys
import time
import numpy as np
import pickle
from tqdm import tqdm
import shutil
import multiprocessing

from mip import Model, xsum, minimize, BINARY, OptimizationStatus

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


def split_list(lst: List, num_chunks: int = 500):
    """
    Splits lst into n equally sized sublists.

    Args:
    - lst: The list to be split.
    - n: The number of sublists to split lst into.

    Returns:
    - A list of sublists, where each sublist is as equal in size as possible.
    """
    k, m = divmod(len(lst), num_chunks)
    return [
        lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_chunks)
    ]


def save_intermediate_results(
    code_configs: List[Dict], chunk_index: int, result_subfolder: str
):
    if not os.path.exists(result_subfolder):
        os.makedirs(result_subfolder)
    file_path = os.path.join(
        result_subfolder, f"codes_with_distance_{chunk_index}.pickle"
    )
    with open(file_path, "wb") as f:
        pickle.dump(code_configs, f)


def load_and_unify_intermediate_results(
    num_known_files, subfolder="intermediate_results_code_search"
):
    unified_configs, files_to_delete = (
        [],
        [],
    )  # List to keep track of files to delete after unification

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
            with open(file_path, "rb") as f:
                configs = pickle.load(f)
                unified_configs.extend(configs)

    # Save the unified results
    unified_file_path = os.path.join(
        subfolder, f"unified_codes_with_distance_{next_file_index}.pickle"
    )
    with open(unified_file_path, "wb") as f:
        pickle.dump(unified_configs, f)
    logging.warning("Saved unified results to: {}".format(unified_file_path))

    # Delete the intermediate files
    for file_path in files_to_delete:
        os.remove(file_path)

    return unified_configs


def calculate_code_distance(code_config: Dict) -> int:
    """
    Calculates and returns the code distance for a given code configuration
    """

    try:
        # computes the minimum Hamming weight of a binary vector x such that
        # stab @ x = 0 mod 2
        # logicOp @ x = 1 mod 2
        # here stab is a binary matrix and logicOp is a binary vector
        def distance_test(stab, logicOp, code_config):
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

            opt_val = sum([x[i].x for i in range(n)])

            return int(opt_val)

        distance = code_config.get("num_phys_qubits")
        hx = code_config.get("hx").toarray()
        lx = code_config.get("lx").toarray()
        for i in range(code_config.get("num_log_qubits")):
            w = distance_test(hx, lx[i, :], code_config)
            distance = min(distance, w)

        code_config["distance"] = distance
        return code_config

    except Exception as e:
        if "Non-optimal solution" not in str(e):
            logging.warning(
                "An error happened in the distance calculation: {}".format(e)
            )
            # Indicate an error occurred
            code_config["distance"] = "Error in code distance calculation: {e}"
        return code_config


def dummy_operation(code_config: List[Dict]):
    """
    Dummy operation to simulate the time it takes to calculate the code distance for a given code configuration
    """
    code_config["distance"] = "dummy_value"
    return code_config


def get_code_distance_parallel(
    code_configs: List[Dict], num_known_files: int, original_subfolder_name: str
):
    # The folder where results will be saved, named after the original subfolder
    result_subfolder = os.path.join(
        "intermediate_results_code_distance_during_ongoing_code_search",
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
            code_configs_with_distance = pool.map(calculate_code_distance, chunk)

        # Save intermediate results in the new subfolder
        save_intermediate_results(
            code_configs=code_configs_with_distance,
            chunk_index=i + 1,
            result_subfolder=result_subfolder,
        )

    logging.warning(
        "------------------ FINISHED CODE DISTANCE CALCULATION ------------------"
    )
    logging.warning(
        f"Distance calculation took: {round((time.time() - start_time) / 3600.0, 2)} hours."
    )

    # Once all chunks are processed, unify the intermediate results and clean up
    try:
        load_and_unify_intermediate_results(num_known_files, subfolder=result_subfolder)
    except Exception as e:
        logging.warning(
            f"An error occurred while unifying the intermediate results: {e}"
        )
        return


def check_for_new_files(directory, known_files):
    """
    Scan the given directory and its subdirectories for new files not listed in known_files.

    Args:
    - directory: Directory to scan for files.
    - known_files: A set of already processed or known file names.

    Returns:
    - A list of new files found in the directory and its subdirectories.
    """
    new_files = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path) and file_name not in known_files:
                new_files.append(file_path)
    return new_files


def main(code_configs_dir: str):
    known_files = set()
    num_known_files = 0

    while True:
        new_files = check_for_new_files(code_configs_dir, known_files)
        if new_files:
            for file_path in new_files:
                if file_path.endswith(".pickle"):
                    original_subfolder_name = os.path.relpath(
                        os.path.dirname(file_path), code_configs_dir
                    )
                    logging.warning(
                        f"Processing file from subfolder: {original_subfolder_name}"
                    )
                    with open(file_path, "rb") as f:
                        code_configs = pickle.load(f)
                    get_code_distance_parallel(
                        code_configs, num_known_files + 1, original_subfolder_name
                    )
                    known_files.add(
                        os.path.relpath(file_path, code_configs_dir)
                    )  # Update known_files with the new file
                    num_known_files += 1
                else:
                    continue
        else:
            logging.warning(
                "No new files found. Waiting for 30 minutes before checking again."
            )

        # Wait for 30 minutes before next check
        time.sleep(1800)


if __name__ == "__main__":
    code_configs_dir = "intermediate_results_code_search"
    main(code_configs_dir)