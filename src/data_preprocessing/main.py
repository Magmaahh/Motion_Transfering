from functions import *

# Main execution loop
if __name__ == "__main__":
    for folder in os.listdir(INPUT_PATH):
        # Retrieve all target sequences from the selected folder
        folder_input_path = os.path.join(INPUT_PATH, folder)
        sequences = get_sequences(folder_input_path)

        # Process each sequence and save the regressed ANNY parameters
        for sequence in sequences:
            # Load the data sample (.npz) containing the SMPL-X parameters to process from the given file
            sequence_input_path = os.path.join(folder_input_path, sequence)
            sequence = np.load(sequence_input_path, allow_pickle=True)

            # Processt the sequence to regress ANNY parameters (phenotypes, local changes and pose parameters)
            params = process_sequence(sequence)

            # Save regressed ANNY parameters
            folder_output_path = os.path.join(OUTPUT_PATH, folder)
            sequence_name = sequence.replace('.npz', '')
            sequence_output_path = os.path.join(folder_output_path, sequence_name)
            os.makedirs(sequence_output_path, exist_ok=True)
            save_parameters(params, sequence_output_path, sequence_name)