import argparse
from soma.io import add_npz_args

from functions import *

# Main execution loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPLX to ANNY Parameter Regression")
    parser.add_argument("--single-input", type=str, help="Path to the single input SMPLX .npz file to process.")
    add_npz_args(parser)
    args = parser.parse_args()

    if args.single_input:
        print(f"-- SINGLE INPUT MODE: {args.single_input} --")
        sequence = np.load(args.single_input, allow_pickle=True)
        params, results = process_sequence(sequence)
        output_folder = os.path.dirname(args.single_input)
        save_parameters(params, output_folder, os.path.basename(args.single_input))
        log_results(os.path.basename(args.single_input), results, output_folder)
    else:
        print(f"-- BATCH MODE: Processing all sequences in {INPUT_PATH} --")
        for folder in os.listdir(INPUT_PATH):
            # Retrieve all target sequences from the selected folder
            folder_input_path = os.path.join(INPUT_PATH, folder)
            files = get_sequences(folder_input_path)

            # Process each sequence and save the regressed ANNY parameters
            for file in files:
                # Load the data sample (.npz) containing the SMPL-X parameters to process from the given file
                sequence_input_path = os.path.join(folder_input_path, file)
                sequence = np.load(sequence_input_path, allow_pickle=True)

                # Process the sequence to regress ANNY parameters (phenotypes, local changes and pose parameters)
                params, results = process_sequence(sequence)

                # Save regressed ANNY parameters
                folder_output_path = os.path.join(OUTPUT_PATH, folder)
                sequence_output_path = os.path.join(folder_output_path, file)
                os.makedirs(sequence_output_path, exist_ok=True)
                save_parameters(params, sequence_output_path, file)
                log_results(file, results, folder_output_path)