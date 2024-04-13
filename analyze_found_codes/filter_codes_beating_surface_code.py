import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
sys.path.append(parent_dir)

# Import the get_net_encoding_rate function from the helper_functions.py file
from helper_functions import (
    extract_file_and_parent_directory,
    load_from_pickle,
    save_as_pickle,
    filter_best_codes_group_by_weight,
)

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def process_pickle_file(file_path, output_dirs):
    codes = load_from_pickle(file_path)

    # Filter codes for: having a distance >= 3, num_log_qubits >= 2, and beating the surface code
    # And group them by weight
    (
        codes_beating_surface_code,
        codes_not_beating_surface_code,
    ) = filter_best_codes_group_by_weight(codes)

    beating_path = os.path.join(
        output_dirs["beating_surface_code"], os.path.basename(file_path)
    )
    not_beating_path = os.path.join(
        output_dirs["NOT_beating_surface_code"], os.path.basename(file_path)
    )

    # Saving the split codes into separate sub-directories, only if non-empty
    if codes_beating_surface_code:
        save_as_pickle(beating_path, codes_beating_surface_code)
    else:
        logging.warning(
            f"NO codes found in {extract_file_and_parent_directory(file_path)} that beat the surface code."
        )
    if codes_not_beating_surface_code:
        save_as_pickle(not_beating_path, codes_not_beating_surface_code)
    else:
        logging.warning(
            f"ALL codes found in {extract_file_and_parent_directory(file_path)}  beat the surface code."
        )


def split_and_save_best_codes(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "slice_" not in os.path.basename(root):
            continue
        # Preparing output directories for categorized codes
        output_dirs = {
            "beating_surface_code": os.path.join(root, "beating_surface_code"),
            "NOT_beating_surface_code": os.path.join(root, "NOT_beating_surface_code"),
        }

        for file_name in files:
            if file_name.startswith(
                "unified_codes_with_distance_"
            ) and file_name.endswith(".pickle"):
                file_path = os.path.join(root, file_name)
                process_pickle_file(file_path, output_dirs)


if __name__ == "__main__":
    # Replace 'your_base_directory_path' with the path to your 'intermediate_results_code_distance_during_ongoing_code_search' directory.
    base_directory_path = "/Users/lukasvoss/Documents/Persönliche Unterlagen/Singapur 2023-2024/03_AStar_KishorBharti/02_Research/ldpc_codes/intermediate_results_code_distance_during_ongoing_code_search"
    split_and_save_best_codes(base_directory_path)
