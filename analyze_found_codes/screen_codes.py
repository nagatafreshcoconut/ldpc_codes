import os
import pickle


def find_highest_values(dictionaries, keys):
    # Initialize dictionaries to hold the highest values for each key
    highest = {key: None for key in keys}
    for d in dictionaries:
        for key in keys:
            if highest[key] is None or d[key] > highest[key][key]:
                highest[key] = d
    return highest


def log_properties(code, property_name):
    print(
        f"Code with l={code['l']}, m={code['m']}, A={code['A_poly_sum']}, B={code['B_poly_sum']} leads to "
        f"[[n={code['num_physical_qubits']}, k={code['num_logical_qubits']}, d={code['distance']}]] "
        f"and encoding_rate={code['encoding_rate']} for highest {property_name}."
    )


def analyze_codes(folder):
    for root, dirs, files in os.walk(folder):
        if os.path.basename(root) == "processed":
            for file in files:
                if file == "processed_consolidated_codes.pickle":
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "rb") as f:
                            codes = pickle.load(f)

                            # Check if the loaded file is empty
                            if not codes:
                                print(f"File {file_path} is empty. Skipping.")
                                continue

                            # Correct dictionary keys
                            for code in codes:
                                code["num_physical_qubits"] = code.pop(
                                    "num_phys_qubits"
                                )
                                code["num_logical_qubits"] = code.pop("k")

                            highest_values = find_highest_values(
                                codes,
                                ["distance", "num_logical_qubits", "encoding_rate"],
                            )

                            log_properties(highest_values["distance"], "distance (d)")
                            log_properties(
                                highest_values["num_logical_qubits"],
                                "number of logical qubits (k)",
                            )
                            log_properties(
                                highest_values["encoding_rate"], "encoding rate"
                            )

                    except (EOFError, pickle.UnpicklingError):
                        print(
                            f"Error reading file {file_path}. It may be corrupted or not a valid pickle file. Skipping."
                        )


if __name__ == "__main__":
    folder_name = "intermediate_results_code_distance_during_ongoing_code_search"
    analyze_codes(folder_name)
