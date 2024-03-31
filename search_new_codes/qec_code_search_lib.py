from typing import List, Dict, Optional
import os
import sys
import io
import time
import numpy as np
from scipy.sparse import csr_matrix
import pickle
from itertools import product
from tqdm import tqdm
from bposd.css import css_code

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


def save_code_configs(my_codes: List[Dict], file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(my_codes, f)


def save_intermediate_results(
    code_configs: List[Dict],
    chunk_index: int,
    slice_identifier: str,
    folder="intermediate_results_code_search",
):
    if not os.path.exists(folder):
        os.makedirs(folder)

    slice_specific_subfolder = os.path.join(folder, slice_identifier)
    if not os.path.exists(slice_specific_subfolder):
        os.makedirs(slice_specific_subfolder)
    file_path = os.path.join(
        slice_specific_subfolder, f"codes_chunk_{chunk_index}.pickle"
    )
    with open(file_path, "wb") as f:
        pickle.dump(code_configs, f)


def load_and_unify_intermediate_results(folder="intermediate_results_code_search"):
    unified_configs = []

    # List all pickle files in the specified folder
    for filename in os.listdir(folder):
        if filename.startswith("codes_chunk_") and filename.endswith(".pickle"):
            file_path = os.path.join(folder, filename)

            # Load the content of each pickle file and extend the master list
            with open(file_path, "rb") as f:
                configs = pickle.load(f)
                unified_configs.extend(configs)

    return unified_configs


def format_slice_identifier(l_value, m_value, weight_value) -> str:
    # Sort the lists to ensure proper ordering
    l_value_sorted = sorted(l_value)
    m_value_sorted = sorted(m_value)
    weight_value_sorted = sorted(weight_value)

    # Format l_value part by joining with commas
    l_part = "".join(map(str, l_value_sorted))

    # Format m_value part by joining with commas
    m_part = "".join(map(str, m_value_sorted))

    # Format weight_value part by joining with commas
    weight_part = "".join(map(str, weight_value_sorted))

    # Combine all parts
    slice_identifier = f"l{l_part}_m{m_part}_weight{weight_part}"
    return slice_identifier


def get_net_encoding_rate(
    k: int,
    n: int,
) -> float:
    return k / (2.0 * n)


def get_valid_powers_for_summands(summand_combo, l, m, range_A, range_B):
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
):
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

def canonical_form(poly_sum):
    # Split the polynomial sum into terms, sort them, and join back into a string
    terms = poly_sum.split(" + ")
    terms.sort()
    return " + ".join(terms)

def build_code(
    l: int,
    m: int,
    x: Dict[int, np.ndarray],
    y: Dict[int, np.ndarray],
    z: Dict[int, np.ndarray],
    summand_combo_A: tuple[str, ...],
    summand_combo_B: tuple[str, ...],
    powers_A: tuple[int],
    powers_B: tuple[int],
    encoding_rate_threshold: Optional[float],
    code_configs: List[Dict],
    existing_codes: set,
):
    A, B = np.zeros((l * m, l * m), dtype=int), np.zeros((l * m, l * m), dtype=int)
    A_poly_sum, B_poly_sum = "", ""

    # Construct A with its summands and powers
    for summand, power in zip(summand_combo_A, powers_A):
        if summand == "x":
            matrix = x[power]
        elif summand == "y":
            matrix = y[power]
        elif summand == "z":
            matrix = z[power]
        A += matrix
        A_poly_sum += f"{summand}{power} + "

    # Construct B with its summands and powers
    for summand, power in zip(summand_combo_B, powers_B):
        if summand == "x":
            matrix = x[power]
        elif summand == "y":
            matrix = y[power]
        elif summand == "z":
            matrix = z[power]
        B += matrix
        B_poly_sum += f"{summand}{power} + "

    A = A % 2
    B = B % 2

    # Remove trailing ' + '
    A_poly_sum = A_poly_sum.rstrip(" + ")
    B_poly_sum = B_poly_sum.rstrip(" + ")

    # Ensure saving identical codes only once: Check the polynomials of A and B (x1 + z2) is the same as (z2 + x1)
    # Create canonical forms for A_poly_sum and B_poly_sum
    A_poly_canonical = canonical_form(A_poly_sum)
    B_poly_canonical = canonical_form(B_poly_sum)
    # Combine the canonical forms to create a unique key for this configuration
    code_key = f"A: {A_poly_canonical}, B: {B_poly_canonical}"
    if code_key in existing_codes:
        # Skip saving this configuration as it's a duplicate
        return

    # Transpose matrices A and B
    AT = np.transpose(A)
    BT = np.transpose(B)

    # Construct matrices hx and hz
    hx = np.hstack((A, B))
    hz = np.hstack((BT, AT))

    # Construct and test the CSS code
    qcode = css_code(hx, hz)  # Define css_code, assuming it's defined elsewhere

    ### Surpress the print output of the css_code.test()
    # Redirect stdout to a dummy StringIO object
    sys.stdout = io.StringIO()

    if qcode.test():  # Define the test method for qcode
        if qcode.K == 0:  # If the code has no logical qubits, skip it
            return
        sys.stdout = original_stdout  # Reset stdout to original value to enable logging
        r = get_net_encoding_rate(int(qcode.K), int(qcode.N))  # Define get_net_encoding_rate
        encoding_rate_threshold = (
            1 / 15 if encoding_rate_threshold is None else encoding_rate_threshold
        )
        code_config = {
            "l": l,
            "m": m,
            "num_phys_qubits": qcode.N,
            "num_log_qubits": qcode.K,
            "lx": csr_matrix(qcode.lx),
            "hx": csr_matrix(hx),
            "encoding_rate": r,
            "encoding_rate_threshold_exceeded": r > encoding_rate_threshold,
            "A_poly_sum": A_poly_sum,
            "B_poly_sum": B_poly_sum,
        }
        code_configs.append(code_config)
        existing_codes.add(code_key)


def search_code_space(
    l_range: list,
    m_range: list,
    weight_range: list,
    power_range_A: list,
    power_range_B: list,
    encoding_rate_threshold: Optional[float],
    slice_identifier: str,
    max_size_mb=1,  # Maximum size for each batch file in MB
):
    """
    Searching the parameter space for good bicycle codes (BC)

    args:
        - l_range: List of possible values for parameter l
        - m_range: List of possible values for parameter m
        - weight_range: List of code weights (= the total number of summands accumulated for both A and B)
        - power_range_A: List of possible values for exponents for terms in A (A is a sum over polynomials in x and y)
        - power_range_B: List of possible values for exponents for terms in B (B is a sum over polynomials in x and y)
        - encoding_rate_threshold (float): the lower bound for codes to be saved for further analysis
    """
    code_configs = []
    chunk_index = 0  # To name the output files uniquely
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes

    for l, m in tqdm(product(l_range, m_range), total=len(l_range) * len(m_range)):
        try:
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

            # Iterate over weights and distribute them across A and B
            for weight in weight_range:
                existing_codes = set()
                for weight_A in range(
                    1, weight
                ):  # Ensure at least one term in A and B # TODO: Could think of also raising to the power of zero leading to identity matrix
                    weight_B = weight - weight_A

                    # Generate all combinations of summands in A and B with their respective weights
                    summands_A = list(product(["x", "y", "z"], repeat=weight_A))
                    summands_B = list(product(["x", "y", "z"], repeat=weight_B))

                    for summand_combo_A, summand_combo_B in product(
                        summands_A, summands_B
                    ):
                        # Check for powers_A
                        # Iterate over power ranges for each summand in A and B
                        for powers_A in get_valid_powers_for_summands(
                            summand_combo_A, l, m, power_range_A, power_range_B
                        ):
                            # Check for powers_B
                            for powers_B in get_valid_powers_for_summands(
                                summand_combo_B, l, m, power_range_A, power_range_B
                            ):
                                try:
                                    build_code(
                                        l,
                                        m,
                                        x,
                                        y,
                                        z,
                                        summand_combo_A,
                                        summand_combo_B,
                                        powers_A,
                                        powers_B,
                                        encoding_rate_threshold,
                                        code_configs,
                                        existing_codes,
                                    )
                                    temp_batch_size = len(pickle.dumps(code_configs))

                                    if temp_batch_size > max_size_bytes:
                                        # Save the current batch and start a new one
                                        logging.warning(
                                            "Saving intermediate results with {} codes.".format(
                                                len(code_configs)
                                            )
                                        )
                                        save_intermediate_results(
                                            code_configs,
                                            chunk_index + 1,
                                            slice_identifier,
                                        )
                                        chunk_index += 1
                                        code_configs = []

                                except Exception as e:
                                    logging.warning(
                                        "An error happened in the code construction: {}".format(
                                            e
                                        )
                                    )
                                    continue

        except Exception as e:
            logging.warning(
                "An error happened in the parameter space search: {}".format(e)
            )
            continue

    # Save any remaining configurations after the loop
    if code_configs:
        save_intermediate_results(code_configs, chunk_index + 1, slice_identifier)


def main(param_space):
    start_time = time.time()

    # Define the specific values for l, m, and weight
    l_value = param_space["l_value"]
    m_value = param_space["m_value"]
    weight_value = param_space["weight_value"]
    power_range_A = param_space["power_range_A"]
    power_range_B = param_space["power_range_B"]

    # Define the slice identifier
    slice_identifier = "slice_" + format_slice_identifier(
        l_value, m_value, weight_value
    )

    # Setup the logging
    logging.warning(
        "------------------ STARTING CODE SEARCH FOR SLICE: {} ------------------".format(
            format_slice_identifier(l_value, m_value, weight_value)
        )
    )

    # Calculate the total number of iterations
    total_iterations = calculate_total_iterations(
        l_value, m_value, weight_value, power_range_A, power_range_B
    )
    logging.warning("Total iterations: {} thousands".format(total_iterations / 1e3))

    # Search for good configurations (since interdependent cannot properly parallelized)
    search_code_space(
        l_range=l_value,
        m_range=m_value,
        weight_range=weight_value,
        power_range_A=power_range_A,
        power_range_B=power_range_B,
        encoding_rate_threshold=1 / 15,
        slice_identifier=slice_identifier,
        max_size_mb=50,  # Maximum size for each batch file in MB
    )

    elapsed_time = round((time.time() - start_time) / 3600.0, 2)
    logging.warning("------------------ FINISHED CODE SEARCH ------------------")
    logging.warning("Elapsed Time: {} hours.".format(elapsed_time))
