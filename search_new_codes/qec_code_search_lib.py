from typing import List, Dict, Optional, Union
import os
import sys
import io
import time
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import galois
from itertools import product
from tqdm import tqdm
from bposd.css import css_code


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import the get_net_encoding_rate function from the helper_functions.py file
from helper_functions import (
    get_net_encoding_rate,
    save_as_pickle,
    get_valid_powers_for_summands,
    canonical_form,
    calculate_total_iterations,
    build_x_y_z_matrices,
    construct_matrix_and_poly_sum,
    get_code_weight,
    get_all_combinations,
    make_code_dim_numerical,
)
from qutrit_css_code import QutritCSSCode


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


class SearchLDPCCodes:

    def __init__(self, param_space: Dict, result_dir: str = "intermediate_results_code_search", encoding_rate_threshold: float = 1/15, max_size_mb_per_saved_file: int = 50):
        self.param_space = param_space
        self.result_dir = result_dir
        self.encoding_rate_threshold = encoding_rate_threshold
        self.max_size_mb = max_size_mb_per_saved_file


    def save_intermediate_results(
        self,
        code_configs: List[Dict],
        chunk_index: int,
        slice_identifier: str,
    ):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        slice_specific_subfolder = os.path.join(self.result_dir, slice_identifier)
        if not os.path.exists(slice_specific_subfolder):
            os.makedirs(slice_specific_subfolder)
        file_path = os.path.join(
            slice_specific_subfolder, f"codes_chunk_{chunk_index}.pickle"
        )
        save_as_pickle(file_path, code_configs)


    def load_and_unify_intermediate_results(self):
        unified_configs = []

        # List all pickle files in the specified folder
        for filename in os.listdir(self.folder):
            if filename.startswith("codes_chunk_") and filename.endswith(".pickle"):
                file_path = os.path.join(self.folder, filename)

                # Load the content of each pickle file and extend the master list
                with open(file_path, "rb") as f:
                    configs = pickle.load(f)
                    unified_configs.extend(configs)

        return unified_configs


    def create_slice_identifier(self, l_value, m_value, weight_value) -> str:
        """
        Formats the given parameters into a slice identifier string.

        Args:
            l_value (list): A list of integers representing the l_value.
            m_value (list): A list of integers representing the m_value.
            weight_value (list): A list of integers representing the weight_value.

        Returns:
            str: The formatted slice identifier string.

        Example:
            >>> self.create_slice_identifier([1, 2, 3], [4, 5, 6], [7, 8, 9])
            'codedim2_l123_m456_weight789'
        """
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
        slice_identifier = f"codedim{self._code_dimension}_l{l_part}_m{m_part}_weight{weight_part}"
        return slice_identifier

    def get_check_matrices_and_code(self, A: np.ndarray, B: np.ndarray, AT: np.ndarray, BT: np.ndarray, code_dimension: Union[int, str] = 2):
            
        if code_dimension == 2 or code_dimension == 'qubit': # Qubit case
            hx = np.hstack((A, B))
            hz = np.hstack((BT, AT))

            qcode = css_code(hx, hz)

        elif code_dimension == 3 or code_dimension == 'qutrit': # Qutrit case
            GFp = galois.GF(int(3))

            # Construct matrices hx and hz
            hx = np.hstack((A, (-B)%3))
            hz = np.hstack((BT, AT))
            
            # Use custom class QutritCSSCode, make the matrices SageMath matrix, automatically changes all entries mod p
            qcode = QutritCSSCode(GFp(hx), GFp(hz))

        else:
            raise NotImplementedError("Only code dimensions 2 and 3 are supported.")

        return hx, hz, qcode

    def build_code(
        self,
        code_dimension : Union[int, str],
        l: int,
        m: int,
        x: Dict[int, np.ndarray],
        y: Dict[int, np.ndarray],
        z: Dict[int, np.ndarray],
        summand_combo_A: tuple[str, ...],
        summand_combo_B: tuple[str, ...],
        powers_A: tuple[int],
        powers_B: tuple[int],
        code_configs: List[Dict],
        existing_codes: set,
    ) -> None:
        """
        Builds a quantum error-correcting (QEC) Low-Density Parity-Check (LDPC) code using the given parameters.

        Args:
            code_dimension (int | str): The dimension of the code (2 for qubit, 3 for qutrit).
            l (int): The value of 'l' parameter.
            m (int): The value of 'm' parameter.
            x (Dict[int, np.ndarray]): A dictionary of matrices for summand 'x' with corresponding powers.
            y (Dict[int, np.ndarray]): A dictionary of matrices for summand 'y' with corresponding powers.
            z (Dict[int, np.ndarray]): A dictionary of matrices for summand 'z' with corresponding powers.
            summand_combo_A (tuple[str, ...]): A tuple of summands for constructing matrix A.
            summand_combo_B (tuple[str, ...]): A tuple of summands for constructing matrix B.
            powers_A (tuple[int]): A tuple of powers corresponding to the summands in summand_combo_A.
            powers_B (tuple[int]): A tuple of powers corresponding to the summands in summand_combo_B.
            code_configs (List[Dict]): A list to store the configurations of the generated codes.
            existing_codes (set): A set to keep track of existing code configurations.

        Returns:
            None: This function appends the codes to the code_configs list.
        """
        A, A_poly_sum = construct_matrix_and_poly_sum(
            l, m, summand_combo_A, powers_A, x, y, z
        )
        B, B_poly_sum = construct_matrix_and_poly_sum(
            l, m, summand_combo_B, powers_B, x, y, z
        )

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
            # Skip this code as it has already been saved before (in a nother permutation of the summands in A and B)
            return

        # Transpose matrices A and B
        AT = np.transpose(A)
        BT = np.transpose(B)

        if code_dimension == 2 or code_dimension == 'qubit':
            hx, hz, qcode = self.get_check_matrices_and_code(A, B, AT, BT, code_dimension)
        else:
            raise NotImplementedError('Currently only the qubit case, so code_dimension=2 is supported.')

        ### Surpress the print output of the css_code.test()
        # Redirect stdout to a dummy StringIO object
        sys.stdout = io.StringIO()

        if qcode.test():  # Define the test method for qcode
            # If the code has no logical qubits or is trivial, skip it
            if qcode.K == 0 or qcode.N == qcode.K:  
                return
            sys.stdout = original_stdout  # Reset stdout to original value to enable logging
            r = get_net_encoding_rate(int(qcode.K), int(qcode.N))  # Define get_net_encoding_rate
            code_config = {
                "code_dimension": code_dimension,
                "l": l,
                "m": m,
                "num_phys_qubits": qcode.N,
                "num_log_qubits": qcode.K,
                "weight": get_code_weight(A_poly_sum, B_poly_sum),
                "lx": csr_matrix(qcode.lx),
                "lz": csr_matrix(qcode.lz),
                "hx": csr_matrix(hx),
                "hz": csr_matrix(hz),
                "encoding_rate": r,
                "encoding_rate_threshold_exceeded": r > self.encoding_rate_threshold,
                "A_poly_sum": A_poly_sum,
                "B_poly_sum": B_poly_sum,
            }
            code_configs.append(code_config)
            existing_codes.add(code_key)


    def search_code_space(
        self,
        code_dimension: Union[int, str],
        l_range: list,
        m_range: list,
        weight_range: list,
        power_range_A: list,
        power_range_B: list,
    ):
        """
        Searching the parameter space for good bicycle codes (BC)

        args:
            - code_dimension (int | str): The dimension of the code (2 for qubit, 3 for qutrit).
            - l_range: List of possible values for parameter l
            - m_range: List of possible values for parameter m
            - weight_range: List of code weights (= the total number of summands accumulated for both A and B)
            - power_range_A: List of possible values for exponents for terms in A (A is a sum over polynomials in x and y)
            - power_range_B: List of possible values for exponents for terms in B (B is a sum over polynomials in x and y)
        """
        code_configs = []
        chunk_index = 0  # To name the output files uniquely
        max_size_bytes = self.max_size_mb * 1024 * 1024  # Convert MB to bytes

        for l, m in tqdm(product(l_range, m_range), total=len(l_range) * len(m_range)):
            try:
                x, y, z = build_x_y_z_matrices(l, m)

                # Iterate over weights and distribute them across A and B
                for weight in weight_range:
                    existing_codes = set()
                    for weight_A in range(
                        1, weight
                    ):  # Ensure at least one term in A and B # TODO: Could think of also raising to the power of zero leading to identity matrix
                        weight_B = weight - weight_A

                        # Generate all combinations of summands in A and B with their respective weights
                        summands_A = get_all_combinations(elements=["x", "y", "z"], repetitions=weight_A)
                        summands_B = get_all_combinations(elements=["x", "y", "z"], repetitions=weight_B)

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
                                        self.build_code(
                                            code_dimension,
                                            l,
                                            m,
                                            x,
                                            y,
                                            z,
                                            summand_combo_A,
                                            summand_combo_B,
                                            powers_A,
                                            powers_B,
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
                                            self.save_intermediate_results(
                                                code_configs,
                                                chunk_index + 1,
                                                self._slice_identifier,
                                            )
                                            chunk_index += 1
                                            code_configs = []

                                    except Exception as e:
                                        logging.warning(
                                            "An error happened in the code construction: {}".format(
                                                e
                                            )
                                        )
                                        raise
                                        # continue

            except Exception as e:
                logging.warning(
                    "An error happened in the parameter space search: {}".format(e)
                )
                raise
                # continue

        # Save any remaining configurations after the loop
        if code_configs:
            self.save_intermediate_results(code_configs, chunk_index + 1, self._slice_identifier)


    def search_codes(self):
        start_time = time.time()

        # Define the specific values for l, m, and weight
        code_dimension = self.param_space["code_dimension"]
        l_value = self.param_space["l_value"]
        m_value = self.param_space["m_value"]
        weight_value = self.param_space["weight_value"]
        power_range_A = self.param_space["power_range_A"]
        power_range_B = self.param_space["power_range_B"]

        # Define the slice identifier
        self._code_dimension = make_code_dim_numerical(code_dimension)

        # Define the slice identifier
        self._slice_identifier = "slice_" + self.create_slice_identifier(
            l_value, m_value, weight_value
        )

        # Setup the logging
        logging.warning(
            "------------------ STARTING CODE SEARCH FOR SLICE: {} ------------------".format(
                self.create_slice_identifier(l_value, m_value, weight_value)
            )
        )

        # Calculate the total number of iterations
        self._total_iterations = calculate_total_iterations(
            l_value, m_value, weight_value, power_range_A, power_range_B
        )
        logging.warning("Total iterations: {} thousands".format(self._total_iterations / 1e3))

        # Search for good configurations (since interdependent cannot properly parallelized)
        self.search_code_space(
            code_dimension=code_dimension,
            l_range=l_value,
            m_range=m_value,
            weight_range=weight_value,
            power_range_A=power_range_A,
            power_range_B=power_range_B,
        )

        elapsed_time = round((time.time() - start_time) / 3600.0, 2)
        logging.warning("------------------ FINISHED CODE SEARCH ------------------")
        logging.warning("Elapsed Time: {} hours.".format(elapsed_time))

    @property
    def code_dimension(self):
        return self.param_space["code_dimension"]
    
    @property
    def l_value(self):
        return self.param_space["l_value"]
    
    @property
    def m_value(self):
        return self.param_space["m_value"]
    
    @property
    def weight_value(self):
        return self.param_space["weight_value"]
    
    @property
    def power_range_A(self):
        return self.param_space["power_range_A"]
    
    @property
    def power_range_B(self):
        return self.param_space["power_range_B"]
    
    @property
    def slice_identifier(self):
        return self._slice_identifier
    
    @property
    def total_iterations(self):
        return self._total_iterations
