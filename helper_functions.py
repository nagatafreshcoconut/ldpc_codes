"""
A set of helper functions that will be used along the whole project value chain.

Author: Lukas Voss, CQT Singapore; lukasvoss@partner.nus.edu.sg
Created: 04th April 2024
"""

from typing import List, Dict, Tuple, Union
import os
import sys
import numpy as np
import pickle
from itertools import product
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

plt.style.use("ggplot")


import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

### General data processing functions


def load_from_pickle(file_path: str):
    """Load data from a pickle file."""
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
    except Exception as e:
        logging.warning("Failed to open file {}".format(file_path))
        logging.warning("Error Message: {}".format(e))
    return data


def save_as_pickle(file_path: str, data) -> None:
    """Save data as a pickle file."""
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def extract_file_and_parent_directory(full_path: str) -> str:
    # Get the file name
    file_name = os.path.basename(full_path)
    # Get the full directory path
    directory_path = os.path.dirname(full_path)
    # Get the parent directory name
    parent_directory = os.path.basename(directory_path)
    # Combine the parent directory and file name
    result = os.path.join(parent_directory, file_name)
    return result


def extract_file_and_parent_directory(full_path: str) -> str:
    # Get the file name
    file_name = os.path.basename(full_path)
    # Get the full directory path
    directory_path = os.path.dirname(full_path)
    # Get the parent directory name
    parent_directory = os.path.basename(directory_path)
    # Combine the parent directory and file name
    result = os.path.join(parent_directory, file_name)
    return result


### Below functions belong to searching new codes


def get_net_encoding_rate(
    k: int,
    n: int,
) -> float:
    return k / (2.0 * n)


def get_valid_powers_for_summands(summand_combo, l, m, range_A, range_B) -> product:
    """
    Generates valid power combinations for a given summand combination,
    respecting the constraints for 'x', 'y', 'z', and the specified ranges for A and B.

    Args:
    - summand_combo: A combination of summands ('x', 'y', 'z').
    - l, m: The limits for 'x' and 'y', respectively. For 'z', the limit is max(l, m).
    - range_A, range_B: Ranges of exponents for terms in A and B to be within.

    Returns:
    - A generator that yields valid combinations of powers for the summands.
    """
    # Define initial power ranges based on summand type
    power_ranges = {
        "x": range(max(1, min(range_A)), min(l, max(range_A))),
        "y": range(max(1, min(range_B)), min(m, max(range_B))),
        "z": range(
            max(1, min(min(range_A), min(range_B))),
            min(max(l, m), max(max(range_A), max(range_B))),
        ),
    }

    # Get the adjusted power range for each summand in the combination
    ranges_for_combo = [power_ranges[summand] for summand in summand_combo]

    # Use product to generate all valid combinations within the specified ranges
    return product(*ranges_for_combo)


def calculate_total_iterations(
    l_range, m_range, weight_range, power_range_A, power_range_B
) -> int:
    total_iterations = 0
    for l, m in product(l_range, m_range):
        for weight in weight_range:
            for weight_A in range(1, weight):  # Ensure at least one term in A and B
                weight_B = weight - weight_A
                summands_A = list(product(["x", "y", "z"], repeat=weight_A))
                summands_B = list(product(["x", "y", "z"], repeat=weight_B))
                for summand_combo_A, summand_combo_B in product(summands_A, summands_B):
                    for powers_A in get_valid_powers_for_summands(
                        summand_combo_A, l, m, power_range_A, power_range_B
                    ):
                        for powers_B in get_valid_powers_for_summands(
                            summand_combo_B, l, m, power_range_A, power_range_B
                        ):
                            total_iterations += 1
    return total_iterations


def canonical_form(poly_sum) -> str:
    # Split the polynomial sum into terms, sort them, and join back into a string
    terms = poly_sum.split(" + ")
    terms.sort()
    return " + ".join(terms)


def build_x_y_z_matrices(
    l: int, m: int
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Builds the matrices x, y, and z based on the given dimensions.

    Parameters:
    l (int): The number of rows in the identity matrix I_ell.
    m (int): The number of columns in the identity matrix I_m.

    Returns:
    Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    A tuple containing three dictionaries:
        - x: A dictionary of matrices x[i] for i in range(l).
        - y: A dictionary of matrices y[j] for j in range(m).
        - z: A dictionary of matrices z[k] for k in range(max(l, m)).

    """
    I_ell = np.identity(l, dtype=int)
    I_m = np.identity(m, dtype=int)
    x, y, z = {}, {}, {}

    # Generate base matrices x and y
    for i in range(l):
        x[i] = np.kron(np.roll(I_ell, i, axis=1), I_m)
    for j in range(m):
        y[j] = np.kron(I_ell, np.roll(I_m, j, axis=1))

    # Create base matrix z
    for k in range(np.max([l, m])):
        z[k] = np.kron(np.roll(I_ell, k, axis=1), np.roll(I_m, k, axis=1))

    return x, y, z


def construct_matrix_and_poly_sum(
    l, m, summand_combo, powers, x: dict, y: dict, z: dict
):
    """
    Constructs a matrix sum and a polynomial sum based on the given summand combination, powers, and matrices.

    Args:
        summand_combo (tuple): A list of summands ("x", "y", or "z") representing the matrices to be used in the sum.
        powers (list): A list of powers corresponding to each summand in the summand_combo.
        x (dict): A dictionary of matrices with keys representing the powers of "x".
        y (dict): A dictionary of matrices with keys representing the powers of "y".
        z (dict): A dictionary of matrices with keys representing the powers of "z".

    Returns:
        tuple: A tuple containing the matrix sum and the polynomial sum.
    """
    matrix_sum = np.zeros((l * m, l * m), dtype=int)
    poly_sum_str = ""
    for summand, power in zip(summand_combo, powers):
        if summand == "x":
            matrix = x[power]
        elif summand == "y":
            matrix = y[power]
        elif summand == "z":
            matrix = z[power]
        matrix_sum += matrix
        poly_sum_str += f"{summand}{power} + "
    return matrix_sum, poly_sum_str


def get_code_weight(A_poly_sum: str, B_poly_sum: str) -> int:
    """
    Calculate the weight of a code.
    The weight of a code is defined as the total number of summands in the A_poly_sum and B_poly_sum of the code.
    Args:
        code (Dict): A dictionary containing the code information.
    Returns:
        int: The weight of the code.
    """
    # Split the sum_expression by '+' and strip whitespace from each summand
    weigth_A = [summand.strip() for summand in A_poly_sum.split("+")]
    weigth_B = [summand.strip() for summand in B_poly_sum.split("+")]
    return sum([len(weigth_A), len(weigth_B)])


def get_all_combinations(elements: List[str], repetitions: int) -> list:
    return list(product(elements, repeat=repetitions))


def make_code_dim_numerical(code_dim: Union[int, str]) -> int:
    if isinstance(code_dim, str):
        if code_dim.lower() == "qubit":
            return 2
        elif code_dim.lower() == "qutrit":
            return 3
    elif isinstance(code_dim, Union[int, float]):
        return int(code_dim)
    else:
        raise ValueError(
            "Invalid code dimension. Must be 'qubit', 'qutrit', or an integer 2 or 3."
        )


### Below functions are used for the code distance calculation


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


### Below functions are used after the distance has been calculated and we prepare codes for the decoding


def sort_list_dicts(list_data: List[Dict], key) -> List[Dict]:
    return sorted(list_data, key=lambda x: x.get(key, 0), reverse=True)


def check_beaten_surface_code_characteristics(
    distance: int, num_phys_qubits: int
) -> bool:
    return (2 * distance**2) > num_phys_qubits


def best_code_conditions(d: dict) -> bool:
    """Checks if a QEC code has better characteristics than the surface code."""
    assert isinstance(d, dict), "Code configuration must be a dictionary."

    distance_summary = d.get("distance_summary")
    # Ensure distance_summary is a dictionary
    if not isinstance(distance_summary, dict):
        return False

    distance = distance_summary.get("distance")
    # distance = d.get("distance")
    num_log_qubits = d.get("num_log_qubits")
    num_phys_qubits = d.get("num_phys_qubits")

    return (
        isinstance(distance, int)
        and distance >= 3
        and isinstance(num_log_qubits, int)
        and num_log_qubits >= 2
        and check_beaten_surface_code_characteristics(distance, num_phys_qubits)
    )


def group_codes_by_weight(codes: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Groups codes by their weight.

    Parameters:
    - codes (List[Dict]): A list of code dictionaries.

    Returns:
    - Dict[int, List[Dict]]: A dictionary where each key is a weight group, and each value is a list of code dictionaries that have this weight.
    """
    grouped_by_weight = {}
    for code in codes:
        if not hasattr(code, "weight"):
            code["weight"] = get_code_weight(code["A_poly_sum"], code["B_poly_sum"])
        weight = code["weight"]
        grouped_by_weight.setdefault(weight, []).append(code)

    return grouped_by_weight


def filter_best_codes_group_by_weight(data: List[Dict]) -> List[Dict]:
    """
    Filters a list of QEC codes based on specific criteria.

    Args:
        data (list): A list of dictionaries representing the codes.

    Returns:
        list: A filtered list of codes sorted by 'encoding_rate' in descending order.
    """
    assert isinstance(data, list), "Input data must be a list of dictionaries."

    best_codes = [d for d in data if best_code_conditions(d)]
    best_codes = sort_list_dicts(best_codes, "encoding_rate")
    print("best codes beating the surface code:", len(best_codes))

    other_codes = [d for d in data if not best_code_conditions(d)]
    other_codes = sort_list_dicts(other_codes, "encoding_rate")
    print("All remaining codes NOT beating surface code:", len(other_codes))

    # Remaining codes that do not meet the criteria
    if len(best_codes) == 0:
        logging.warning("No codes beat the surface code.")
        return ({}, group_codes_by_weight(data))
    elif len(best_codes) == len(data):
        logging.warning("All codes beat the surface code.")
        return (group_codes_by_weight(best_codes), {})

    return (group_codes_by_weight(best_codes), group_codes_by_weight(other_codes))


### Below functions are used for the decoding


def rebuild_hz_from_hx(hx: np.ndarray) -> csr_matrix:
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


def add_hz_to_code(grouped_codes: Dict[int, List[Dict]]):
    # Add stabilizer check Hz to the code configs if not already present
    for _, codes in grouped_codes.items():
        # for property_name, codes in properties.items():
        for code in codes:
            # Assuming 'hx' is stored directly in each code dictionary
            # and that 'hx' is already an np.array or similar that supports slicing
            hx = (
                code["hx"].toarray()
                if isinstance(code["hx"], csr_matrix)
                else code["hx"]
            )  # Convert to np.array if saved as sparse matrix
            code["hz"] = rebuild_hz_from_hx(hx)
    return grouped_codes


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


def get_decoder_marker(decoder_name: str) -> str:
    decoder_markers = {
        "BeliefPropagationOSDDecoder": "BPOSD",
        "Z3Decoder": "Z3",
    }
    return decoder_markers.get(decoder_name, decoder_name)
