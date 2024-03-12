import os
import sys
import shutil
import pickle
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

def delete_entries(list_data):
    # Remove specific keys
    for d in list_data:
        for key in ['hx', 'hz', 'lx', 'A', 'B']:
            d.pop(key, None)
            
    return list_data


def sort_list_dicts(list_data, key):
    return sorted(list_data, key=lambda x: x.get(key, 0), reverse=True)

def process_data(data):
        
        # Filter dictionaries
        filtered_data = [
            d for d in data 
            if isinstance(d.get('distance'), int) and d.get('distance') > 0
            and isinstance(d.get('num_log_qubits'), int) and d.get('num_log_qubits') > 0
        ]
        
        # Sort the list by 'encoding_rate' in descending order
        data_sorted_by_encoding_rate = sort_list_dicts(filtered_data, 'encoding_rate')
         
        return data_sorted_by_encoding_rate

def process_directories(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        logging.warning('Searching in subdirectory: {}'.format(subdir))
        processed_data_aggregate = []
        for file in files:
            if file.endswith('.pickle'):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, 'rb') as file:
                        data = pickle.load(file)
                except Exception as e:
                    logging.warning('Failed to open file {}'.format(file_path))
                    logging.warning('Error Message: {}'.format(e))
                    continue
                processed_data = process_data(data)
                processed_data_aggregate.extend(processed_data)
        
        processed_data_aggregate = process_data(processed_data_aggregate)
     
        # Create a 'processed' subfolder if it doesn't exist
        processed_subfolder = os.path.join(subdir, 'processed')
        if not os.path.exists(processed_subfolder):
            os.makedirs(processed_subfolder)
        
        # Save the processed data in the 'processed' subfolder
        processed_file_path = os.path.join(processed_subfolder, f'processed_consolidated_codes.pickle')
        with open(processed_file_path, 'wb') as out_file:
            pickle.dump(processed_data_aggregate, out_file)
        logging.warning('Successfully saved consolidated and processed codes: {}'.format(len(processed_data_aggregate)))

def delete_processed_folders(main_directory):
    for subdir, dirs, files in os.walk(main_directory):
        for dir_name in dirs:
            if dir_name == 'processed':
                # Construct the full path to the 'processed' folder
                processed_folder_path = os.path.join(subdir, dir_name)
                
                # Remove the 'processed' folder and all its contents
                shutil.rmtree(processed_folder_path)
                
                print(f"Deleted: {processed_folder_path}")

if __name__ == '__main__':
    root_dir = 'intermediate_results_code_distance_during_ongoing_code_search'
    delete_processed_folders(root_dir)
    process_directories(root_dir)