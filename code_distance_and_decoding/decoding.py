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
from css_code import CSSCode
from z3_decoder import Z3Decoder

from panqec.codes import Color666PlanarCode
from panqec.bpauli import get_effective_error
from panqec.codes import Toric2DCode

from panqec.config import CODES, ERROR_MODELS, DECODERS
CODES['CSSCode'] = CSSCode
DECODERS['Z3Decoder'] = Z3Decoder

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


def save_results_as_pickle(code_configs_decoded, result_subfolder):
    if not os.path.exists(result_subfolder):
        os.makedirs(result_subfolder)
    file_path = os.path.join(result_subfolder, "codes_decoded.pickle")
    with open(file_path, "wb") as f:
        pickle.dump(code_configs_decoded, f)


def load_codes_from_pickle(file_path):
    with open(file_path, "rb") as f:
        grouped_codes = pickle.load(f)

    # Add stabilizer check Hz to the code configs
    for weight, properties in grouped_codes.items():
        for property_name, codes in properties.items():
            for code in codes:
                # Assuming 'hx' is stored directly in each code dictionary
                # and that 'hx' is already an np.array or similar that supports slicing
                hx = code['hx'].toarray() if not isinstance(code['hx'], np.ndarray) else code['hx'] # Convert to np.array if saved as sparse matrix
                code['hz'] = rebuild_hz_from_hx(hx)
    return grouped_codes


def load_from_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def plot_results(effective_error, error_rate):
    effective_error_all = np.any(
        effective_error[:, :, :], axis=2
    )  ##check if any logical error has occurred
    avg_effective_error_all = np.mean(
        effective_error_all, axis=1
    )  ##error occuring for any logical observable

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(error_rate, avg_effective_error_all)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate")
    ax.set_ylabel("logical error rate")
    plt.show()


def get_input_data(
    label,
    parameters_code,
    error_model,
    decoder,
    parameters_decoder,
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
        "error_rate": np.linspace(
            min_error_rate, max_error_rate, n_error_rate
        ).tolist(),  # List of physical error rates
        "rounds": rounds,  ##number of rounds of error correction (on same instance)
        "n_trials": n_trials,  ##number of repetitions of QEC to get statistics
        "filename": filename,  ##name to save
    }

def perform_decoding(args):
    code, weight, property_name, index = args
    rng = np.random.default_rng()

    name_code, decoder, error_model, size_code = get_code_details()
    (
        n_trials,
        rounds,
        min_error_rate,
        max_error_rate,
        n_error_rate,
    ) = get_decoding_details()

    Hx = code["hx"].toarray() if not isinstance(code["hx"], np.ndarray) else code["hx"]
    Hz = code["hz"].toarray() if not isinstance(code["hz"], np.ndarray) else code["hz"]

    parameters_decoder={}
    if decoder == "BeliefPropagationOSDDecoder":
        n_qubits = np.shape(Hx)[1]
        osd_order_limit = min(n_qubits-np.linalg.matrix_rank(Hx), n_qubits-np.linalg.matrix_rank(Hz))
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

    input_data = get_input_data(
        label,
        parameters_code,
        error_model,
        decoder,
        parameters_decoder,
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

    # Only keep the relevant results
    # code["decoding_results"] = {
    #     "effective_error_list": result_Dict["effective_error_list"],
    #     "error_rate": result_Dict["error_rate"],
    # }

    return {
        "effective_error_list": csr_matrix(result_Dict["effective_error_list"]),
        "error_rate": result_Dict["error_rate"],
    }


def perform_decoding_parallel(original_subfolder_name, grouped_code_configs):
    result_subfolder = os.path.join(
        "intermediate_results_decoding",
        original_subfolder_name,
    )

    # TODO: Use knowledge about structure of the saved dicts
    num_processes = multiprocessing.cpu_count()

    start_time = time.time()
    logging.warning("------------------ START DECODING ------------------")
    logging.warning(f"Number of processes: {num_processes}")
    logging.warning(f"Number of code configurations: {len(grouped_code_configs)}")

    args_list = []
    for weight, properties in grouped_code_configs.items():
        for property_name, codes in properties.items():
            for index, code in enumerate(codes):
                args_list.append((code, weight, property_name, index))


    # Decode codes in parallel and collect results
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        decoding_results = pool.map(perform_decoding, args_list)

    # Integrate decoding results back into the original structure
    for decoding_result in decoding_results:
        grouped_code_configs[weight][property_name][index]['decoding_results'] = decoding_result

    save_results_as_pickle(grouped_code_configs, result_subfolder)
    logging.warning('Saved decoding results to: {}'.format(result_subfolder))
    logging.warning("------------------ END DECODING ------------------")
    logging.warning(f"Decoding took {round(time.time() - start_time, 0)} seconds")


def main(code_configs_dir):
    for root, dirs, files in os.walk(code_configs_dir):
        # Check if 'codes_with_highest_properties_to_be_decoded' is in the current root's subdirectories
        if 'codes_with_highest_properties_to_be_decoded' in dirs:
            target_dir = os.path.join(root, 'codes_with_highest_properties_to_be_decoded')
            # Process each pickle file within the target directory
            for filename in os.listdir(target_dir):
                if filename.endswith(".pickle"):
                            
                    file_path = os.path.join(target_dir, filename)
                    print('Decoding this file:', file_path)
                    
                    try:
                        grouped_code_configs = load_codes_from_pickle(file_path)
                        # grouped_code_configs = grouped_code_configs[:10]  # For testing purposes
                        original_subfolder_name = os.path.basename(os.path.dirname(target_dir))
                        perform_decoding_parallel(original_subfolder_name, grouped_code_configs)
                    except Exception as e:
                        logging.error(f"Error for {file_path}: {e}")
                        raise
                        continue

def get_code_details():
    name_code = "Toric"
    decoder = "BeliefPropagationOSDDecoder"
    error_model = "PauliErrorModel"
    size_code = 6
    return name_code, decoder, error_model, size_code


def get_decoding_details():
    n_trials = int(1e3)  # repetititons to get statistics
    rounds = 1  # rounds of Error correction applied

    min_error_rate = 5*1e-3
    max_error_rate = 1e-1

    n_error_rate = 20

    return n_trials, rounds, min_error_rate, max_error_rate, n_error_rate

if __name__ == "__main__":
    code_configs_dir = "intermediate_results_code_distance_during_ongoing_code_search"
    main(code_configs_dir)
