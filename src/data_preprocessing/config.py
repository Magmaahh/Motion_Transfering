import torch

# Paths configuration
INPUT_PATH = "data/smplx_data" # Path to the directory containing the smplx sequences to process
MODELS_PATH = "src/data_preprocessing/models" # Path to the directory containing models
OUTPUT_PATH = "data/anny_data" # Path to to the directory where ANNY parameters are stored
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants configuration
COMPARE = True # Whether to compare the resulting meshes with the original smplx ones
COMPARE_FRAME = 40 # Frequency of frames to compare (if COMPARE is enabled)
DISPLACEMENT_THRESHOLD = 0.03 # Displacement threshold for motion-diverse frame sampling (e.g., 0.02 means 2% of body height)
TARGET_RATIO = 0.10 # Desired fraction of frames to keep for phenotype regression (e.g., 0.1 means keep 10% of frames)
VERBOSE = True # Whether to print PVE values during processing