"""
Created on Sun Feb 25 10:48:43 2024

Run PanQEC for error correction
Can take in arbitrary CSS codes

@author: Tobias Haug @TII, tobias.haug@u.nus.edu
@modified by: Lukas Voss @CQT, lukasvoss@partner.nus.edu.sg

"""

from typing import List, Dict
import os
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
from panqec.codes import Color666PlanarCode
from panqec.bpauli import get_effective_error
from panqec.codes import Toric2DCode
from panqec.config import CODES, ERROR_MODELS, DECODERS

print(CODES.keys())


import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def parse_code_dict(code_dict):
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


def parse_error_model_dict(noise_dict):
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


def parse_decoder_dict(decoder_dict, code, error_model, error_rate):
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


def run_QEC_once(code, decoder, error_model, error_rate, rng, rounds):
    total_error = 0
    for k in range(rounds):  ##rounds of error correction
        error = error_model.generate(code, error_rate=error_rate, rng=rng)
        error = error + total_error  ##add uncorrected errors from previous rounds
        syndrome = code.measure_syndrome(error)
        correction = decoder.decode(syndrome)
        total_error = (correction + error) % 2

    effective_error = get_effective_error(total_error, code.logicals_x, code.logicals_z)
    codespace = code.in_codespace(total_error)
    success = bool(np.all(effective_error == 0)) and codespace

    return effective_error, codespace, success


def run_QEC_batch(input_data, rng):
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
    code = parse_code_dict(code_dict)

    n_qubits = code.n
    n_logicals = code.k

    ##setup error model
    error_model = parse_error_model_dict(noise_dict)

    ##setup decoders
    decoder_list = []
    for k in range(n_error_params):
        error_r = error_rate[k]
        decoder = parse_decoder_dict(decoder_dict, code, error_model, error_r)
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
            effective_error, codespace, success = run_QEC_once(
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

    full_filename = filename + "_" + timestr + ".pcl"
    fullDict = [input_data, resultDict]
    outfile = open(full_filename, "wb")
    pickle.dump(fullDict, outfile)
    outfile.close()

    return (effective_error_list, codespace_list, avg_walltime, full_filename)


def rebuild_hz_from_hx(hx):
    # Number of columns in hx represents the total 'n'
    # and rows in hx are 'n/2', so we split hx vertically in the middle.
    n_cols = hx.shape[1]
    half_n_cols = n_cols // 2

    # Split Hx back into A and B based on its shape (n/2) x n.
    A = hx[:, :half_n_cols]
    B = hx[:, half_n_cols:]

    # Transpose A and B to get AT and BT.
    # Note: In this scenario, transposing doesn't change the shape from (n/2) x (n/2) to any other shape
    # because A and B are not square matrices, but the step is kept for consistency with H_z construction logic.
    AT, BT = A.T, B.T

    # Construct Hz by horizontally stacking BT and AT.
    # Since A and B are (n/2) x (n/2), their transposes are also (n/2) x (n/2), making Hz (n/2) x n.
    hz = np.hstack((BT, AT))

    return csr_matrix(hz)


def save_intermediate_results(
    code_configs: List[Dict], chunk_index: int, result_subfolder: str
):
    if not os.path.exists(result_subfolder):
        os.makedirs(result_subfolder)
    file_path = os.path.join(result_subfolder, f"codes_decoded_{chunk_index}.pickle")
    with open(file_path, "wb") as f:
        pickle.dump(code_configs, f)


def load_and_unify_intermediate_results(
    subfolder="intermediate_results_code_distance_during_ongoing_code_search",
):
    unified_configs, files_to_delete = (
        [],
        [],
    )  # List to keep track of files to delete after unification

    # Count existing unified pickle files to determine the index for the new file
    existing_unified_files = [
        f for f in os.listdir(subfolder) if f.startswith("unified_codes_decoded_")
    ]
    next_file_index = len(existing_unified_files) + 1

    for filename in os.listdir(subfolder):
        if filename.startswith("codes_decoded") and filename.endswith(".pickle"):
            file_path = os.path.join(subfolder, filename)
            # Add this file to the list of files to delete
            files_to_delete.append(file_path)

            # Load the content of each pickle file and extend the unified list
            with open(file_path, "rb") as f:
                configs = pickle.load(f)
                unified_configs.extend(configs)

    # Save the unified results
    unified_file_path = os.path.join(
        subfolder, f"unified_codes_decoded_{next_file_index}.pickle"
    )
    with open(unified_file_path, "wb") as f:
        pickle.dump(unified_configs, f)
    logging.warning("Saved unified results to: {}".format(unified_file_path))

    # Delete the intermediate files
    for file_path in files_to_delete:
        os.remove(file_path)


def load_codes_from_pickle(file_path):
    with open(file_path, "rb") as f:
        code_configs = pickle.load(f)

    # Add stabilizer check Hz to the code configs
    for code in code_configs:
        code["hz"] = rebuild_hz_from_hx(code["hx"].toarray())
    return code_configs


def load_from_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


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


def plot_results(effective_error, error_rate):
    effective_error_all = np.any(
        effective_error[:, :, :], axis=2
    )  ##check if any logical error has occurred
    avg_effective_error_all = np.mean(
        effective_error_all, axis=1
    )  ##error occuring for any logical observable

    plt.plot(error_rate, avg_effective_error_all)
    plt.set_xscale("log")
    plt.set_yscale("log")
    plt.ylabel("logical error rate")
    plt.xlabel("physical error rate")
    plt.show()


def get_input_data(
    label,
    parameters_code,
    error_model,
    decoder,
    min_error_rate,
    max_error_rate,
    n_error_rate,
    rounds,
    n_trials,
    filename,
):
    return {
        "label": label,  # Can be any name you want
        "code": {
            "name": "StabilizerCode",  # Class name of the code
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
        "decoder": {"name": decoder, "parameters": {}},
        "error_rate": np.linspace(
            min_error_rate, max_error_rate, n_error_rate
        ).tolist(),  # List of physical error rates
        "rounds": rounds,  ##number of rounds of error correction (on same instance)
        "n_trials": n_trials,  ##number of repetitions of QEC to get statistics
        "filename": filename,  ##name to save
    }


def get_code_details():
    name_code = "Toric"
    decoder = "MatchingDecoder"
    error_model = "PauliErrorModel"
    size_code = 6
    return name_code, decoder, error_model, size_code


def get_decoding_details():
    n_trials = 1e4  # repetititons to get statistics
    rounds = 4  # rounds of Error correction applied

    min_error_rate = 1e-2
    max_error_rate = 1e-1

    n_error_rate = 20

    return n_trials, rounds, min_error_rate, max_error_rate, n_error_rate


def perform_decoding(code):
    rng = np.random.default_rng()

    name_code, decoder, error_model, size_code = get_code_details()
    (
        n_trials,
        rounds,
        min_error_rate,
        max_error_rate,
        n_error_rate,
    ) = get_decoding_details()

    Hx, Hz = code["hx"], code["hz"]

    parameters_code = {
        "Hx_in": Hx,
        "Hz_in": Hz,
        "name": name_code + "_L" + str(size_code),
        "L_in": size_code,
    }

    label = name_code + "_L" + str(size_code) + "_" + decoder
    filename = "results/" + label

    input_data = get_input_data(
        label,
        parameters_code,
        error_model,
        decoder,
        min_error_rate,
        max_error_rate,
        n_error_rate,
        rounds,
        n_trials,
        filename,
    )
    effective_error_list, codespace_list, avg_walltime, full_filename = run_QEC_batch(
        input_data, rng
    )

    input_data, result_Dict = load_from_pickle(full_filename)

    code["decoding_results"] = result_Dict

    return code


def perform_decoding_parallel(original_subfolder_name, code_configs):
    result_subfolder = os.path.join(
        "intermediate_results_decoding",
        original_subfolder_name,
    )

    # Determine the number of processes to use
    num_processes = multiprocessing.cpu_count()
    num_chunks = len(code_configs) // 200
    if num_chunks < 1:
        num_chunks = 1
    chunked_list = split_list(code_configs, num_chunks)  # Split the list into chunks

    start_time = time.time()
    logging.warning("------------------ START DECODING ------------------")
    logging.warning(f"Number of processes: {num_processes}")
    logging.warning(f"Number of code configurations: {len(code_configs)}")

    for i, chunk in enumerate(tqdm(chunked_list)):
        logging.warning(f"Number of codes in Chunk {i+1}: {len(chunk)}")

        with multiprocessing.Pool(processes=num_processes) as pool:
            code_configs_decoded = pool.map(perform_decoding, chunk)

        # Save intermediate results in the new subfolder
        save_intermediate_results(
            code_configs=code_configs_decoded,
            chunk_index=i + 1,
            result_subfolder=result_subfolder,
        )
    # Once all chunks are processed, unify the intermediate results and clean up
    try:
        load_and_unify_intermediate_results(subfolder=result_subfolder)
    except Exception as e:
        logging.warning(
            f"An error occurred while unifying the intermediate results: {e}"
        )
        return


def main(code_configs_dir):
    for root, dirs, files in os.walk(code_configs_dir):
        original_subfolder_name = os.path.basename(root)
        for file in files:
            if file.startswith("unified_codes_with_distance") and file.endswith(
                ".pickle"
            ):
                file_path = os.path.join(root, file)
                code_configs = load_codes_from_pickle(file_path)
                try:
                    perform_decoding_parallel(original_subfolder_name, code_configs)
                except Exception as e:
                    logging.error(f"Error in {file_path}: {e}")
                    continue


if __name__ == "__main__":
    code_configs_dir = "intermediate_results_code_distance_during_ongoing_code_search"
    main(code_configs_dir)
