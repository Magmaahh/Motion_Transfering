import torch

# -- Paths configuration --
# General paths
ASSETS_PATH = "src/data_preprocessing/assets" # Path to the directory containing assets used for processing
MODELS_PATH = "src/data_preprocessing/assets/models" # Path to the directory containing models used for processing

# SMPLX to ANNY pipeline paths
INPUT_PATH = "data/smplx_data/gendered" # Path to the directory containing the gendered smplx sequences to process
OUTPUT_PATH = "data/anny_data" # Path to to the directory where ANNY parameters are stored

# SMPLX to SOMA pipeline paths
SOMA_INPUT_PATH = "data/smplx_data/neutral" # Path to the directory containing the neutral smplx sequences to process
SOMA_OUTPUT_PATH = "data/soma_data" # Path to to the directory where SOMA data is stored

# -- Constants configuration --
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPARE = True # Whether to compare the resulting meshes with the original smplx ones
COMPARE_FRAME = 100 # Frequency of frames to compare (if COMPARE is enabled)
DISPLACEMENT_THRESHOLD = 0.03 # Displacement threshold for motion-diverse frame sampling (e.g., 0.02 means 2% of body height)
TARGET_RATIO = 0.10 # Desired fraction of frames to keep for phenotype regression (e.g., 0.1 means keep 10% of frames)
VERBOSE = True # Whether to print PVE values during processing