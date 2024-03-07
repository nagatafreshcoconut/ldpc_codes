import sys
import time
from qec_code_search_lib import (
    search_code_space, 
    calculate_total_iterations
)

# Setup logging
import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
    

if __name__ == '__main__':

    l_range_identifier = 5  # Example, adjust as needed
    m_range_identifier = 5  # Example, adjust as needed

    l_value = range(l_range_identifier, l_range_identifier + 1)
    m_value = range(m_range_identifier, m_range_identifier + 1)
    
    start_time = time.time()
    # Define the specific values for l, m, and weight
    weight_value = range(5, 6)
    slice_identifier = f"slice_l{l_value.start}-{l_value.stop}_m{m_value.start}-{m_value.stop}_weight_{weight_value.start}-{weight_value.stop}"

    # Define the power ranges for summands in A and B
    # Adjust these ranges as per the specific code you're trying to reproduce
    power_range_A = range(1, 7)  # Example range, adjust as needed
    power_range_B = range(1, 7)  # Example range, adjust as needed

    logging.warning('------------------ STARTING CODE SEARCH FOR SLICE: l_{}-{}, m_{}-{}, weight_{}{}------------------'.format(l_value.start, l_value.stop, m_value.start, m_value.stop, weight_value.start, weight_value.stop))

    # Calculate the total number of iterations
    total_iterations = calculate_total_iterations(l_value, m_value, weight_value, power_range_A, power_range_B)
    logging.warning('Total iterations: {} thousands'.format(total_iterations / 1e3))

    # # Search for good configurations (since interdependent cannot properly parallelized)
    search_code_space(
        l_range=l_value, 
        m_range=m_value, 
        weight_range=weight_value, 
        power_range_A=power_range_A, 
        power_range_B=power_range_B,
        encoding_rate_threshold=1/15,
        slice_identifier=slice_identifier,
        max_size_mb=50, # Maximum size for each batch file in MB
    )

    elapsed_time = round((time.time() - start_time) / 3600.0, 2)
    logging.warning('------------------ FINISHED CODE SEARCH ------------------')
    logging.warning('Elapsed Time: {} hours.'.format(elapsed_time))