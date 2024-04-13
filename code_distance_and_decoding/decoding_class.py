from typing import List, Dict
import os
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use("ggplot")
from scipy.sparse import csr_matrix
import pickle
import datetime
import multiprocessing
from tqdm import tqdm

# from panqec.codes.base._stabilizer_code import StabilizerCode
# from panqec.codes.CSS._css import CSSCode
# from panqec.decoders.z3_decoder._z3decoder import Z3Decoder
from css_code import CSSCode
from z3_decoder import Z3Decoder

from panqec.codes import Color666PlanarCode
from panqec.bpauli import get_effective_error
from panqec.codes import Toric2DCode
from panqec.config import CODES, ERROR_MODELS, DECODERS

CODES["CSSCode"] = CSSCode
DECODERS["Z3Decoder"] = Z3Decoder
# print(DECODERS['Z3Decoder'])

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
sys.path.append(parent_dir)

# Import the get_net_encoding_rate function from the helper_functions.py file
from helper_functions import (
    load_from_pickle,
    save_as_pickle,
    add_hz_to_code,
    get_decoder_marker,
)

### Surpress the print output of the css_code.test() ###
original_stdout = sys.stdout

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


class CustomDecoder:
    """
    A custom decoder class for processing quantum error correction codes, specifically
    designed to handle decoding algorithms, error models, and code configurations.

    This class supports the initialization of code configurations from directory paths,
    parsing of code, error model, and decoder details from dictionaries, and execution
    of quantum error correction (QEC) simulations in batch mode. The class enables
    the parallel processing of multiple decoding tasks and provides functionality to
    save the results.

    Attributes:
        code_configs_dir (str): Directory containing the pickle files with the codes to be decoded.
        code_details (Dict): Details of the quantum error correction code including
                             the code type, parameters, and other relevant information.
        decoding_details (Dict): Details of the decoding process including error rates,
                                 the number of trials, rounds of correction, and other
                                 parameters relevant to the simulation.

    Methods:
        parse_code_dict(code_dict): Parses a dictionary to initialize a quantum code object.
        parse_error_model_dict(noise_dict): Parses a dictionary to initialize an error model object.
        parse_decoder_dict(decoder_dict, code, error_model, error_rate): Parses a dictionary to
            initialize a decoder object with specified code, error model, and error rate.
        run_QEC_once(code, decoder, error_model, error_rate, rng, rounds): Performs one quantum error
            correction simulation and returns the result.
        run_QEC_batch(input_data, rng): Runs a batch of quantum error correction simulations based on
            input data and random number generator (rng).
        save_results_as_pickle(code_configs_decoded, result_subfolder): Saves decoding results into a
            pickle file in the specified subfolder.
        get_input_data(...): Generates input data for a quantum error correction simulation based on specified
            parameters.
        perform_decoding(args): Decodes a given code configuration using specified decoding parameters.
        perform_decoding_parallel(original_subfolder_name, grouped_code_configs): Performs parallel decoding
            of multiple code configurations.
        start_decoding(code_configs_dir): Starts the decoding process for code configurations located in the
            specified directory.
    """

    def __init__(
        self, code_configs_dir: str, code_details: Dict, decoding_details: Dict
    ):
        """
        Args:
            code_configs_dir (str): The directory path for the codes that shall be decoded.
            code_details (Dict): A dictionary containing the details of the code.
            decoding_details (Dict): A dictionary containing the details of the decoding process.
        """
        self.code_configs_dir = code_configs_dir
        self.code_details = code_details
        self.decoding_details = decoding_details

    def parse_code_dict(self, code_dict):
        code_name = code_dict["name"]
        code_params = []
        if "parameters" in code_dict:
            code_params = code_dict["parameters"]
        code_class = CODES[code_name]
        if isinstance(code_params, dict):
            code = code_class(**code_params)  # type: ignore
        else:
            code = code_class(*code_params)  # type: ignore
        return code

    def parse_error_model_dict(self, noise_dict):
        error_model_name = noise_dict["name"]
        error_model_params = []
        if "parameters" in noise_dict:
            error_model_params = noise_dict["parameters"]
        error_model_class = ERROR_MODELS[error_model_name]
        if isinstance(error_model_params, dict):
            error_model = error_model_class(**error_model_params)
        else:
            error_model = error_model_class(*error_model_params)
        return error_model

    def parse_decoder_dict(self, decoder_dict, code, error_model, error_rate):
        decoder_name = decoder_dict["name"]
        decoder_class = DECODERS[decoder_name]
        decoder_params: dict = {}
        if "parameters" in decoder_dict:
            decoder_params = decoder_dict["parameters"]
        else:
            decoder_params = {}

        decoder_params["code"] = code
        decoder_params["error_model"] = error_model
        decoder_params["error_rate"] = error_rate

        decoder = decoder_class(**decoder_params)
        return decoder

    def run_QEC_once(self, code, decoder, error_model, error_rate, rng, rounds):
        total_error = 0
        for k in range(rounds):  ##rounds of error correction
            error = error_model.generate(code, error_rate=error_rate, rng=rng)
            error = error + total_error  ##add uncorrected errors from previous rounds
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(syndrome)
            total_error = (correction + error) % 2

        effective_error = get_effective_error(
            total_error, code.logicals_x, code.logicals_z
        )
        codespace = code.in_codespace(total_error)
        success = bool(np.all(effective_error == 0)) and codespace

        return effective_error, codespace, success

    def run_QEC_batch(self, input_data, rng):
        current_date = datetime.datetime.now()
        timestr = time.strftime("%Y%m%d-%H%M%S")

        error_rate = input_data["error_rate"]
        code_dict = input_data["code"]
        noise_dict = input_data["error_model"]
        decoder_dict = input_data["decoder"]
        n_trials = input_data["n_trials"]
        rounds = input_data["rounds"]
        filename = input_data["filename"]

        n_error_params = len(error_rate)

        ##setup code
        code = self.parse_code_dict(code_dict)

        n_qubits = code.n
        n_logicals = code.k

        ##setup error model
        error_model = self.parse_error_model_dict(noise_dict)

        ##setup decoders
        decoder_list = []
        for k in range(n_error_params):
            error_r = error_rate[k]
            decoder = self.parse_decoder_dict(decoder_dict, code, error_model, error_r)
            decoder_list.append(decoder)

        effective_error_list = np.zeros(
            [n_error_params, n_trials, 2 * n_logicals], dtype=int
        )
        codespace_list = np.zeros([n_error_params, n_trials], dtype=bool)
        walltime_list = [[] for k in range(n_error_params)]

        for rep in range(n_trials):
            if n_trials >= 10 and rep % (n_trials // 10) == 0:
                print(((rep + 1) // (n_trials // 10)) * 10, "% done")

            for k in range(n_error_params):
                start_time = datetime.datetime.now()
                error_r = error_rate[k]
                decoder = decoder_list[k]
                effective_error, codespace, _ = self.run_QEC_once(
                    code, decoder, error_model, error_r, rng, rounds
                )
                effective_error_list[k, rep, :] = effective_error
                codespace_list[k, rep] = codespace
                finish_time = datetime.datetime.now() - start_time
                walltime_list[k].append(finish_time)

        avg_walltime = [np.mean(walltime_list[k]) for k in range(n_error_params)]
        total_time = datetime.datetime.now() - current_date

        resultDict = {
            "effective_error_list": effective_error_list,
            "codespace_list": codespace_list,
            "current_date": current_date,
            "avg_walltime": avg_walltime,
            "error_rate": error_rate,
            "n_qubits": n_qubits,
            "n_logicals": n_logicals,
            "total_time": total_time,
        }

        full_filename = filename + "_" + timestr + ".pickle"
        fullDict = [input_data, resultDict]
        save_as_pickle(full_filename, fullDict)

        return (effective_error_list, codespace_list, avg_walltime, full_filename)

    def save_results_as_pickle(self, code_configs_decoded, result_subfolder):
        if not os.path.exists(result_subfolder):
            os.makedirs(result_subfolder)
        decoder_marker = get_decoder_marker(self.decoder)
        file_path = os.path.join(
            result_subfolder, f"codes_decoded_{decoder_marker}.pickle"
        )
        save_as_pickle(file_path, code_configs_decoded)
        logging.warning(f"Decoding results saved to {file_path}")

    def get_input_data(
        self,
        label,
        parameters_code,
        error_model,
        decoder,
        parameters_decoder,
        error_rate,
        rounds,
        n_trials,
        filename,
    ):
        return {
            "label": label,  # Can be any name you want
            "code": {
                "name": "CSSCode",  # Class name of the code
                "parameters": parameters_code,
            },
            "error_model": {
                "name": error_model,  # Class name of the error model
                "parameters": {  # Ratios of X, Y and Z errors
                    "r_x": 1 / 3,
                    "r_y": 1 / 3,
                    "r_z": 1 / 3,
                },
            },
            "decoder": {
                "name": decoder,
                "parameters": parameters_decoder,
            },
            "error_rate": list(error_rate),  # List of physical error rates
            "rounds": rounds,  ##number of rounds of error correction (on same instance)
            "n_trials": n_trials,  ##number of repetitions of QEC to get statistics
            "filename": filename,  ##name to save
        }

    def perform_decoding(self, args):
        code, weight, index = args
        rng = np.random.default_rng()

        (
            name_code,
            decoder,
            error_model,
            size_code,
            n_trials,
            rounds,
            error_rate,
        ) = (
            self.name_code,
            self.decoder,
            self.error_model,
            self.size_code,
            self.n_trials,
            self.rounds,
            self.error_rate,
        )

        logging.warning("BEFORE: Type of Hx: {}".format(type(code["hx"])))
        logging.warning("BEFORE: Type of Hz: {}".format(type(code["hz"])))
        Hx = code["hx"].toarray() if isinstance(code["hx"], csr_matrix) else code["hx"]
        Hz = code["hz"].toarray() if isinstance(code["hz"], csr_matrix) else code["hz"]

        logging.warning("AFTER: Type of Hx: {}".format(type(Hx)))
        logging.warning("AFTER: Type of Hz: {}".format(type(Hz)))

        parameters_decoder = {}
        if decoder == "BeliefPropagationOSDDecoder":
            n_qubits = np.shape(Hx)[1]
            osd_order_limit = min(
                n_qubits - np.linalg.matrix_rank(Hx),
                n_qubits - np.linalg.matrix_rank(Hz),
            )
            osd_order = min(10, osd_order_limit)
            parameters_decoder["osd_order"] = osd_order

        parameters_code = {
            "Hx_in": Hx,
            "Hz_in": Hz,
            "name": name_code + "_L" + str(size_code),
            "L_in": size_code,
        }

        label = name_code + "_L" + str(size_code) + "_" + decoder
        filename = "results/" + label

        input_data = self.get_input_data(
            label,
            parameters_code,
            error_model,
            decoder,
            parameters_decoder,
            error_rate,
            rounds,
            n_trials,
            filename,
        )
        (
            effective_error_list,
            codespace_list,
            avg_walltime,
            full_filename,
        ) = self.run_QEC_batch(input_data, rng)

        input_data, result_Dict = load_from_pickle(full_filename)

        # Only keep the relevant results
        code["decoding_results"] = {
            "effective_error_list": result_Dict["effective_error_list"],
            "error_rate": result_Dict["error_rate"],
        }

        return code

    def perform_decoding_parallel(self, original_subfolder_name, grouped_code_configs):
        result_subfolder = os.path.join(
            "intermediate_results_decoding",
            original_subfolder_name,
        )
        logging.warning("Result Subfolder: {}".format(result_subfolder))

        num_processes = multiprocessing.cpu_count()

        start_time = time.time()
        logging.warning("------------------ START DECODING ------------------")
        logging.warning(f"Number of processes: {num_processes}")
        logging.warning(f"Number of code configurations: {len(grouped_code_configs)}")

        args_list = []

        for weight, codes in grouped_code_configs.items():
            for index, code in enumerate(codes):
                args_list.append((code, weight, index))

        # Decode codes in parallel and collect results
        # with multiprocessing.Pool(processes=num_processes) as pool:
        #     decoding_results = pool.map(self.perform_decoding, args_list)

        for weight, codes in grouped_code_configs.items():
            for index, code in enumerate(codes):
                args = (code, weight, index)  # Prepare arguments for perform_decoding
                try:
                    decoded_code = self.perform_decoding(
                        args
                    )  # Get decoding result as a dict
                except Exception as e:
                    logging.error(f"Error for code {index}: {e}")
                    # raise
                    continue

                # Directly integrate the decoding result into the code configuration
                grouped_code_configs[weight][index]["decoding_results"] = decoded_code[
                    "decoding_results"
                ]
                logging.warning("Finished decoding a code configuration")

        logging.warning("Finished decoding all code configurations")

        # for ind, result in enumerate(decoding_results):
        #     # Use the same index to access the corresponding task information in args_list
        #     _, weight, index = args_list[ind]  # Correctly unpack the task information

        #     # Assign the decoding result to the correct location
        #     grouped_code_configs[weight][index]['decoding_results'] = result

        self.save_results_as_pickle(grouped_code_configs, result_subfolder)
        logging.warning("------------------ END DECODING ------------------")
        logging.warning(f"Decoding took {round(time.time() - start_time, 0)} seconds")

    def start_decoding(self):
        # Ensure the function `load_codes_from_pickle` and `perform_decoding_parallel` are defined.

        for root, dirs, _ in os.walk(self.code_configs_dir, topdown=True):
            for dir_name in dirs:
                # Check if 'beating_surface_code' is one of the subdirectories in the current directory
                if dir_name == "beating_surface_code":
                    target_dir = os.path.join(root, dir_name)
                    print("Target directory:", target_dir)

                    # Process each pickle file within the target directory
                    for filename in os.listdir(target_dir):
                        if filename.endswith(".pickle"):
                            file_path = os.path.join(target_dir, filename)
                            print("Decoding this file:", file_path)

                            try:
                                # Assuming `load_codes_from_pickle` loads pickle files and returns their content
                                grouped_code_configs = load_from_pickle(file_path)
                                grouped_code_configs = add_hz_to_code(
                                    grouped_code_configs
                                )

                                for weight, codes in grouped_code_configs.items():
                                    codes = [
                                        code
                                        for code in codes
                                        if code["num_phys_qubits"]
                                        != code["num_log_qubits"]
                                    ]
                                    # grouped_code_configs[weight] = codes[2:3]  # Limiting to 2 codes for testing

                                for weight, codes in grouped_code_configs.items():
                                    logging.warning(
                                        f"Weight: {weight}, Number of codes: {len(codes)}"
                                    )

                                # Assuming the directory name immediately before 'beating_surface_code' is required
                                original_subfolder_name = os.path.basename(root)

                                # Assuming `perform_decoding_parallel` is your decoding function
                                self.perform_decoding_parallel(
                                    original_subfolder_name, grouped_code_configs
                                )

                            except Exception as e:
                                logging.error(f"Error for {file_path}: {e}")
                                # Removed 'raise' to continue processing other files even if an error occurs
                                # raise
                                continue

    @property
    def decoder(self):
        return self.code_details["decoder"]

    @property
    def error_model(self):
        return self.code_details["error_model"]

    @property
    def name_code(self):
        return self.code_details["name_code"]

    @property
    def size_code(self):
        return self.code_details["size_code"]

    @property
    def n_trials(self):
        return self.decoding_details["n_trials"]

    @property
    def rounds(self):
        return self.decoding_details["rounds"]

    @property
    def min_error_rate(self):
        return self.decoding_details["min_error_rate"]

    @property
    def max_error_rate(self):
        return self.decoding_details["max_error_rate"]

    @property
    def n_error_rate(self):
        return self.decoding_details["n_error_rate"]

    @property
    def error_rate(self):
        return self.decoding_details["error_rate"]
