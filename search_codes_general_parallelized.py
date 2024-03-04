from typing import List, Dict, Optional
import os
import sys
import io
import time
import numpy as np
import pickle
from itertools import product
from tqdm import tqdm
import multiprocessing

from mip import Model, xsum, minimize, BINARY, OptimizationStatus
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

def save_code_configs(
        my_codes: List[Dict], 
        file_name: str
    ):
    with open(file_name, 'wb') as f:
        pickle.dump(my_codes, f)

def save_intermediate_results(
        code_configs: List[Dict], 
        chunk_index: int, 
        folder='intermediate_results_code_search'
    ):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, f'codes_chunk_{chunk_index}.pickle')
    with open(file_path, 'wb') as f:
        pickle.dump(code_configs, f)

def load_and_unify_intermediate_results(
    folder='intermediate_results_code_search'
):
    unified_configs = []
    
    # List all pickle files in the specified folder
    for filename in os.listdir(folder):
        if filename.startswith('codes_chunk_') and filename.endswith('.pickle'):
            file_path = os.path.join(folder, filename)
            
            # Load the content of each pickle file and extend the master list
            with open(file_path, 'rb') as f:
                configs = pickle.load(f)
                unified_configs.extend(configs)
                
    return unified_configs

def get_net_encoding_rate(
        k: int,
        n: int,
) -> float:
    return k / (2.*n)

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
        'x': range(max(1, min(range_A)), min(l, max(range_A))),
        'y': range(max(1, min(range_B)), min(m, max(range_B))),
        'z': range(max(1, min(min(range_A), min(range_B))), min(max(l, m), max(max(range_A), max(range_B))))
    }

    # Get the adjusted power range for each summand in the combination
    ranges_for_combo = [power_ranges[summand] for summand in summand_combo]

    # Use product to generate all valid combinations within the specified ranges
    return product(*ranges_for_combo)

def calculate_total_iterations(l_range, m_range, weight_range, power_range_A, power_range_B):
    total_iterations = 0
    for l, m in product(l_range, m_range):
        for weight in weight_range:
            for weight_A in range(1, weight):  # Ensure at least one term in A and B
                weight_B = weight - weight_A
                summands_A = list(product(['x', 'y', 'z'], repeat=weight_A))
                summands_B = list(product(['x', 'y', 'z'], repeat=weight_B))
                for summand_combo_A, summand_combo_B in product(summands_A, summands_B):
                    for powers_A in get_valid_powers_for_summands(summand_combo_A, l, m, power_range_A, power_range_B):
                        for powers_B in get_valid_powers_for_summands(summand_combo_B, l, m, power_range_A, power_range_B):
                            total_iterations += 1
    return total_iterations

def build_code(
        l: int, 
        m: int,
        x: Dict[int, np.ndarray],
        y: Dict[int, np.ndarray], 
        z: Dict[int, np.ndarray], 
        summand_combo_A: List[str], 
        summand_combo_B: List[str], 
        powers_A: List[int], 
        powers_B: List[int],
        encoding_rate_threshold: Optional[float],
        code_configs: List[Dict],
    ):
    
    A, B = np.zeros((l*m, l*m), dtype=int), np.zeros((l*m, l*m), dtype=int)
    A_poly_sum, B_poly_sum = '', ''

    # Construct A with its summands and powers
    for summand, power in zip(summand_combo_A, powers_A):
        if summand == 'x':
            matrix = x[power]
        elif summand == 'y':
            matrix = y[power]
        elif summand == 'z':
            matrix = z[power]
        A += matrix
        A_poly_sum += f"{summand}{power} + "

    # Construct B with its summands and powers
    for summand, power in zip(summand_combo_B, powers_B):
        if summand == 'x':
            matrix = x[power]
        elif summand == 'y':
            matrix = y[power]
        elif summand == 'z':
            matrix = z[power]
        B += matrix
        B_poly_sum += f"{summand}{power} + "

    A = A % 2
    B = B % 2

    # Remove trailing ' + '
    A_poly_sum = A_poly_sum.rstrip(' + ')
    B_poly_sum = B_poly_sum.rstrip(' + ')

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
        sys.stdout = original_stdout  # Reset stdout to original value to enable logging
        r = get_net_encoding_rate(qcode.K, qcode.N)  # Define get_net_encoding_rate
        encoding_rate_threshold = 1/15 if encoding_rate_threshold is None else encoding_rate_threshold
        code_config = {
            'l': l,
            'm': m,
            'num_phys_qubits': qcode.N,
            'num_log_qubits': qcode.K,
            'lx': qcode.lx,
            'hx': hx,
            'hz': hz,
            'k': qcode.lz.shape[0], 
            'encoding_rate': r,
            'encoding_rate_threshold_exceeded': r > encoding_rate_threshold,
            'A': A,
            'B': B,
            'A_poly_sum': A_poly_sum,
            'B_poly_sum': B_poly_sum
        }
        code_configs.append(code_config)

def search_codes_general(
        l_range: range, 
        m_range: range, 
        weight_range: range, 
        power_range_A: range, 
        power_range_B: range, 
        encoding_rate_threshold: Optional[float],
        max_size_mb=1  # Maximum size for each batch file in MB
    ):
    """
    Searching the parameter space for good bicycle codes (BC)

    args:
        - l_range: Range of possible values for parameter l
        - m_range: Range of possible values for parameter m
        - weight_range: Range of code weights (= the total number of summands accumulated for both A and B)
        - power_range_A: Range of possible values for exponents for terms in A (A is a sum over polynomials in x and y)
        - power_range_B: Range of possible values for exponents for terms in B (B is a sum over polynomials in x and y)
        - encoding_rate_threshold (float): the lower bound for codes to be saved for further analysis
    """
    code_configs = []
    chunk_index = 0  # To name the output files uniquely
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes

    for l, m in tqdm(product(l_range, m_range), total=len(l_range)*len(m_range)):
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
                for weight_A in range(1, weight):  # Ensure at least one term in A and B # TODO: Could think of also raising to the power of zero leading to identity matrix
                    weight_B = weight - weight_A

                    # Generate all combinations of summands in A and B with their respective weights
                    summands_A = list(product(['x', 'y', 'z'], repeat=weight_A))
                    summands_B = list(product(['x', 'y', 'z'], repeat=weight_B))

                    for summand_combo_A, summand_combo_B in product(summands_A, summands_B):
                        # Check for powers_A
                        # Iterate over power ranges for each summand in A and B
                        for powers_A in get_valid_powers_for_summands(summand_combo_A, l, m, power_range_A, power_range_B):
                            # Check for powers_B
                            for powers_B in get_valid_powers_for_summands(summand_combo_B, l, m, power_range_A, power_range_B):
                                try: 
                                    build_code(l, m, x, y, z, summand_combo_A, summand_combo_B, powers_A, powers_B, encoding_rate_threshold, code_configs)
                                    temp_batch_size = len(pickle.dumps(code_configs))

                                    if temp_batch_size > max_size_bytes:
                                        # Save the current batch and start a new one
                                        print(len(code_configs))
                                        save_intermediate_results(code_configs, chunk_index+1)
                                        chunk_index += 1
                                        code_configs = []
                                
                                except Exception as e:
                                    logging.warning('An error happened in the code construction: {}'.format(e))
                                    continue
                                
        except Exception as e:
            logging.warning('An error happened in the parameter space search: {}'.format(e))
            continue
    
    # Save any remaining configurations after the loop
    if code_configs:
        save_intermediate_results(code_configs, chunk_index+1)


def calculate_code_distance(
        code_config: Dict
    ) -> int:
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
            wstab = np.max([np.sum(stab[i,:]) for i in range(m)])
            # weight of the logical operator
            wlog = np.count_nonzero(logicOp)
            # how many slack variables are needed to express orthogonality constraints modulo two
            num_anc_stab = int(np.ceil(np.log2(wstab)))
            num_anc_logical = int(np.ceil(np.log2(wlog)))
            # total number of variables
            num_var = n + m*num_anc_stab + num_anc_logical

            model = Model()
            model.verbose = 0
            x = [model.add_var(var_type=BINARY) for i in range(num_var)]
            model.objective = minimize(xsum(x[i] for i in range(n)))

            # orthogonality to rows of stab constraints
            for row in range(m):
                weight = [0]*num_var
                supp = np.nonzero(stab[row,:])[0]
                for q in supp:
                    weight[q] = 1
                cnt = 1
                for q in range(num_anc_stab):
                    weight[n + row*num_anc_stab +q] = -(1<<cnt)
                    cnt+=1
                model+= xsum(weight[i] * x[i] for i in range(num_var)) == 0

            # odd overlap with logicOp constraint
            supp = np.nonzero(logicOp)[0]
            weight = [0]*num_var
            for q in supp:
                weight[q] = 1
            cnt = 1
            for q in range(num_anc_logical):
                weight[n + m*num_anc_stab +q] = -(1<<cnt)
                cnt+=1
            model+= xsum(weight[i] * x[i] for i in range(num_var)) == 1

            model.optimize()
            if model.status != OptimizationStatus.OPTIMAL: # If not optimal, record the status and raise an exception
                code_config['distance'] = f'Non-optimal solution: {model.status}'
                raise Exception('Non-optimal solution: {}'.format(model.status))

            opt_val = sum([x[i].x for i in range(n)])

            return int(opt_val)
    

        distance = code_config.get('num_phys_qubits')
        hx = code_config.get('hx')
        lx = code_config.get('lx')
        for i in range(code_config.get('num_log_qubits')):
            w = distance_test(hx, lx[i, :], code_config)
            distance = min(distance, w)

        code_config['distance'] = distance
        return code_config
    
    except Exception as e:
        logging.warning('An error happened in the distance calculation: {}'.format(e))
        # Indicate an error occurred
        code_config['distance'] = 'Error'
        return code_config
    
def split_list(lst: List, num_chunks: int):
    """
    Splits lst into n equally sized sublists.

    Args:
    - lst: The list to be split.
    - n: The number of sublists to split lst into.

    Returns:
    - A list of sublists, where each sublist is as equal in size as possible.
    """
    k, m = divmod(len(lst), num_chunks)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]
    

def get_code_distance_parallel(code_configs):
    # Determine the number of processes to use
    num_processes = multiprocessing.cpu_count()
    chunked_list = split_list(code_configs, 1000) # Split the list into 1000 chunks
    
    start_time = time.time()
    logging.warning('------------------ START CODE DISTANCE CALCULATION ------------------')
    for i, chunk in enumerate(chunked_list):
        logging.warning('Start Code Distance Calculation for Chunk {}: with {} codes'.format(i+1, len(chunk)))

        with multiprocessing.Pool(processes=num_processes) as pool:
            # Map the calculate_code_distance function across all configurations
            # The pool.map function will automatically split the iterable into chunks and assign them to the processes
            code_configs_with_distance = pool.map(calculate_code_distance, chunk)

        logging.warning('Finished Code Distance Calculation for Chunk {}'.format(i+1))
        # Save intermedite results
        save_intermediate_results(code_configs=code_configs_with_distance, chunk_index=i+1, folder='intermediate_results_code_distance')
        logging.warning('Saved intermediate code distance results for Chunk {}'.format(i+1))

    logging.warning('------------------ FINISHED CODE DISTANCE CALCULATION ------------------')
    logging.warning('Distance calculation took: {} hours.'.format(round((time.time() - start_time) / 3600.0, 2)))


if __name__ == '__main__':

    start_time = time.time()
    logging.warning('------------------ STARTING CODE SEARCH ------------------')

    # Define the specific values for l, m, and weight
    l_value = range(6, 7) # only the value 6
    m_value = range(6, 7) # only the value 6
    weight_value = range(6, 7) # only the value 6

    # Define the power ranges for summands in A and B
    # Adjust these ranges as per the specific code you're trying to reproduce
    power_range_A = range(2, 4)  # Example range, adjust as needed
    power_range_B = range(2, 3)  # Example range, adjust as needed

    ### TEST VALUES ###
    # Define the specific values for l, m, and weight
    l_value = range(2, 4) # only the value 6
    m_value = range(2, 3) # only the value 6
    weight_value = range(4, 5) # only the value 6

    # Define the power ranges for summands in A and B
    # Adjust these ranges as per the specific code you're trying to reproduce
    power_range_A = range(1, 4)  # Example range, adjust as needed
    power_range_B = range(1, 4)  # Example range, adjust as needed

    # Calculate the total number of iterations
    total_iterations = calculate_total_iterations(l_value, m_value, weight_value, power_range_A, power_range_B)
    logging.warning('Total iterations: {}'.format(total_iterations))

    # # Search for good configurations (since interdependent cannot properly parallelized)
    search_codes_general(
        l_range=l_value, 
        m_range=m_value, 
        weight_range=weight_value, 
        power_range_A=power_range_A, 
        power_range_B=power_range_B,
        encoding_rate_threshold=1/15,
        max_size_mb=50 # Maximum size for each batch file in MB
    )

    # Load and unify all intermediate results
    try:
        unified_code_configs_no_distance = load_and_unify_intermediate_results(folder='intermediate_results_code_search')
        # Save all code configurations before their distance was calculated
        save_code_configs(unified_code_configs_no_distance, 'codes_no_distance.pickle')
        logging.warning("Total codes saved: {}".format(len(unified_code_configs_no_distance)))
        logging.warning('Saved all code configurations before their distance was calculated.')
        logging.warning('Difference in codes saved vs. total iterations: {}'.format(len(unified_code_configs_no_distance) - total_iterations))

    except Exception as e:
        logging.warning('An error happened in the loading and unification of intermediate results: {}'.format(e))
        logging.warning('Probably the combined size of the intermediate results is too large to be combined into one pickle.')

    
    # Parallel calculation of code distances
    # get_code_distance_parallel(unified_code_configs_no_distance)
    # # Save final results
    try:
        unified_code_configs_with_distance = load_and_unify_intermediate_results(folder='intermediate_results_code_distance')
        # Save all code configurations after their distance was calculated
        save_code_configs(unified_code_configs_with_distance, 'codes_with_distance.pickle')
        logging.warning('Saved all code configurations after their distance was calculated.')
    except Exception as e:
        logging.warning('An error happened in the loading and unification of intermediate results: {}'.format(e))
        logging.warning('Probably the combined size of the intermediate results is too large to be combined into one pickle.')

    elapsed_time = round((time.time() - start_time) / 3600.0, 2)
    logging.warning('------------------ FINISHED CODE SEARCH ------------------')
    logging.warning('Elapsed Time: {} hours.'.format(elapsed_time))