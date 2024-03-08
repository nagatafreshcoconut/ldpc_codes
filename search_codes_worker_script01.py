from qec_code_search_lib import main

"""
Param Config 1: {'l_value': [2], 'm_value': [2, 3], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 2: {'l_value': [2], 'm_value': [4, 5], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 3: {'l_value': [2], 'm_value': [6, 7], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 4: {'l_value': [2, 3], 'm_value': [8, 2], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 5: {'l_value': [3], 'm_value': [3, 4], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 6: {'l_value': [3], 'm_value': [5, 6], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 7: {'l_value': [3], 'm_value': [8, 7], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 8: {'l_value': [4], 'm_value': [2, 3], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 9: {'l_value': [4], 'm_value': [4, 5], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 10: {'l_value': [4], 'm_value': [6, 7], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 11: {'l_value': [4, 5], 'm_value': [8, 2], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 12: {'l_value': [5], 'm_value': [3, 4], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 13: {'l_value': [5], 'm_value': [5, 6], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 14: {'l_value': [5], 'm_value': [8, 7], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 15: {'l_value': [6], 'm_value': [2, 3], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 16: {'l_value': [6], 'm_value': [4, 5], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 17: {'l_value': [6], 'm_value': [6, 7], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 18: {'l_value': [6, 7], 'm_value': [8, 2], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 19: {'l_value': [7], 'm_value': [3, 4], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 20: {'l_value': [7], 'm_value': [5, 6], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 21: {'l_value': [7], 'm_value': [8, 7], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 22: {'l_value': [8], 'm_value': [2, 3], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 23: {'l_value': [8], 'm_value': [4, 5], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 24: {'l_value': [8], 'm_value': [6, 7], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
Param Config 25: {'l_value': [8], 'm_value': [8], 'weight_value': [4, 5, 6], 'power_range_A': [1, 2, 3, 4, 5, 6, 7], 'power_range_B': [1, 2, 3, 4, 5, 6, 7]}
"""

if __name__ == '__main__':

    param_space = {
        'l_value': [2], 
        'm_value': [2, 3], 
        'weight_value': [4, 5, 6], 
        'power_range_A': [1, 2, 3, 4, 5, 6, 7], 
        'power_range_B': [1, 2, 3, 4, 5, 6, 7]
    }
    main(param_space)