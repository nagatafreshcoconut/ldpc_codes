"""
Script to analyze the codes found during the code search process and save the top codes for decoding.

Author: Lukas Voss @CQT, lukasvoss@partner.nus.edu.sg
Created on: 28 March 2024
"""

import os
import pickle
from datetime import datetime

def calculate_weight(code):
    """
    Calculate the weight of a quantum error correction code based on its A and B polynomial terms.
    
    Parameters:
    - code (dict): A dictionary representing a quantum error correction code, including 'A_poly_sum' and 'B_poly_sum' keys.

    Returns:
    - int: The weight of the code, calculated as the sum of the number of terms in 'A_poly_sum' and 'B_poly_sum'.
    """
    a_terms = code['A_poly_sum'].count('+') + 1
    b_terms = code['B_poly_sum'].count('+') + 1
    return a_terms + b_terms

def find_best_values(codes, keys):
    """
    For each weight group, find up to the top three codes based on specified properties.
    
    Parameters:
    - codes (dict of list): A dictionary grouping codes by their weight, with each key mapping to a list of code dictionaries.
    - keys (list): A list of strings representing the keys (properties) of the codes to compare.

    Returns:
    - dict: A nested dictionary where each outer key is a weight group, and each inner key maps to a list of up to three dictionaries with the highest values for the specified key.
    """
    grouped_highest = {}
    for weight, group in codes.items():
        highest = {key: [] for key in keys}
        for d in group:
            try:
                for key in keys:
                    highest[key].append(d)
                    # Ensure we're sorting in a way that higher values are considered better
                    highest[key].sort(key=lambda x: x[key], reverse=True)
                    highest[key] = highest[key][:5]  # Keep only the top 5 values
            except Exception as e:
                continue
        grouped_highest[weight] = highest
    return grouped_highest

def clean_codes(codes):
    """
    Filter out codes that do not meet certain criteria (non-zero logical qubits and distance higher than 3).

    Parameters:
    - codes (list): A list of dictionaries, each representing a quantum error correction code.

    Returns:
    - list: A filtered list of dictionaries for codes that meet the criteria.
    """
    filtered_codes = []
    for code in codes:
        distance = code['distance']
        
        # Check if distance is an integer and meets the criteria
        if isinstance(distance, int) and distance >= 3:
            filtered_codes.append(code)
        # Check if distance is a dict and has a valid integer value that meets the criteria
        elif isinstance(distance, dict) and isinstance(distance.get('distance'), int) and distance['distance'] >= 3:
            filtered_codes.append(code)
    
    return filtered_codes

def log_properties(grouped_best_values):
    """
    Log the properties of the top codes for each weight group and property.

    Parameters:
    - grouped_best_values (dict): A nested dictionary with weight groups as keys and properties as sub-keys, mapping to lists of code dictionaries.
    """
    for weight, properties in grouped_best_values.items():
        print(f"\nWeight Group: {weight}")
        for property_name, codes in properties.items():
            print(f"Top codes for {property_name}:")
            for code in codes:
                print(f"    Code with l={code['l']}, m={code['m']}, A={code['A_poly_sum']}, B={code['B_poly_sum']} leads to [[n={code['num_phys_qubits']}, k={code['num_log_qubits']}, d={code['distance']}]] and encoding_rate={code['encoding_rate']}.")

def save_as_pickle(grouped_best_values, output_folder):
    """
    Save the structured dictionary of grouped best values to a pickle file.

    Parameters:
    - grouped_best_values (dict): The dictionary to save.
    - output_folder (str): The directory path where the pickle file should be saved.
    """
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    filename = f"grouped_best_{timestamp}.pickle"
    output_path = os.path.join(output_folder, filename)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Check if the dictionary is empty
    if not any(grouped_best_values.values()):  # This checks if there's any non-empty value in the dictionary
        print(f"No data to save for {filename}. Skipping.")
        return
    with open(output_path, 'wb') as f:
        pickle.dump(grouped_best_values, f)

def ensure_unique_distances_across_categories(grouped_codes, min_unique_distances=5):
    """
    Adjusts the grouped codes to ensure at least min_unique_distances unique distances across all categories per weight.

    Parameters:
    - grouped_codes (dict): A dictionary grouping codes by their weight, each key mapping to lists of code dictionaries across categories.
    - min_unique_distances (int): Minimum number of unique distance values required across categories.

    Returns:
    - dict: Adjusted nested dictionary with codes ensuring diversity in 'distance' values across categories per weight.
    """
    adjusted_grouped_codes = {}
    for weight, categories in grouped_codes.items():
        # Flatten all codes from all categories, keeping their original category for later reassignment
        all_codes_flat = [(code, category) for category in categories for code in grouped_codes[weight][category]]

        # Filter for unique distances
        unique_distances = {}
        for code, category in all_codes_flat:
            dist = code['distance'] if isinstance(code['distance'], int) else code['distance']['distance']
            if dist not in unique_distances:
                unique_distances[dist] = (code, category)
        
        # Select up to min_unique_distances codes ensuring unique distances
        selected_codes = list(unique_distances.values())[:min_unique_distances]
        
        # Rebuild the categories with the selected codes, ensuring no duplication of distances
        rebuilt_categories = {key: [] for key in categories.keys()}
        for code, category in selected_codes:
            if category in rebuilt_categories:
                rebuilt_categories[category].append(code)

        adjusted_grouped_codes[weight] = rebuilt_categories

    return adjusted_grouped_codes


def analyze_codes(folder, dir_save_best_codes):
    """
    Analyze codes from pickle files in a specified folder, grouping them by weight, finding the top three codes for specified properties, and saving the results.

    Parameters:
    - folder (str): The directory path to search for pickle files.
    - dir_save_best_codes (str): The directory path where the results should be saved.
    """
    for root, dirs, files in os.walk(folder):
        if dir_save_best_codes in os.path.relpath(root, folder).split(os.path.sep):
            continue  # Skip the results directory
        for file in files:
            file_path = os.path.join(root, file)
            print(f"\nAnalyzing file {file_path}")
            try:
                with open(file_path, "rb") as f:
                    codes = pickle.load(f)
                
                if not codes:
                    print("File is empty. Skipping.")
                    continue

                codes = clean_codes(codes)
                grouped_codes = {}
                for code in codes:
                    weight = calculate_weight(code)
                    if weight not in grouped_codes:
                        grouped_codes[weight] = []
                    grouped_codes[weight].append(code)
                # Ensure at least 5 unique distances across categories for each weight
                adjusted_grouped_codes = ensure_unique_distances_across_categories(grouped_codes, min_unique_distances=5)


                grouped_best_values = find_best_values(
                    grouped_codes,
                    ["distance", "num_log_qubits", "encoding_rate"],
                )

                log_properties(grouped_best_values)  # Log the properties of the top codes

                save_as_pickle(grouped_best_values, os.path.join(root, dir_save_best_codes))

            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error reading file {file_path}: {e}. Skipping.")
                continue

if __name__ == "__main__":
    dir_code_input = "intermediate_results_code_distance_during_ongoing_code_search"
    dir_save_best_codes = "codes_best_best_properties_to_be_decoded"
    analyze_codes(dir_code_input, dir_save_best_codes)
