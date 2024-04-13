from code_distance_calculation import CodeDistanceCalculator

if __name__ == "__main__":
    
    found_codes_dir = "intermediate_results_code_search"
    exclude_found_codes_dirs = ['slice_l8_m45_weight456', 'slice_l8_m67_weight456']
    decoded_codes_dir = "intermediate_results_code_distance_during_ongoing_code_search"
    
    distance_calculator = CodeDistanceCalculator(
        found_codes_dir=found_codes_dir,
        exclude_dirs=exclude_found_codes_dirs,
        decoded_codes_dir=decoded_codes_dir
    )
    print(distance_calculator.exclude_dirs)
