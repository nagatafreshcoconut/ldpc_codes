import numpy as np
import sys
from decoding_class import CustomDecoder

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

if __name__ == "__main__":
    code_details = {
        "name_code": "Toric",
        "decoder": "BeliefPropagationOSDDecoder",
        "error_model": "PauliErrorModel",
        "size_code": 6,
    }
    decoding_details = {
        "n_trials": int(1e3),  # repetititons to get statistics
        "rounds": 1,  # rounds of Error correction applied
        "min_error_rate": 1e-2,  # 1e-4
        "max_error_rate": 1e-1,  # 5e-2
        "n_error_rate": 20,
    }
    decoding_details["error_rate"] = np.logspace(
        np.log10(decoding_details["min_error_rate"]),
        np.log10(decoding_details["max_error_rate"]),
        decoding_details["n_error_rate"],
    ).tolist()
    code_configs_dir = "/Users/lukasvoss/Documents/Persönliche Unterlagen/Singapur 2023-2024/03_AStar_KishorBharti/02_Research/ldpc_codes/intermediate_results_code_distance_during_ongoing_code_search"

    decoder = CustomDecoder(code_configs_dir, code_details, decoding_details)
    logging.warning("Using decoder: {}".format(decoder.decoder))
    decoder.start_decoding()
