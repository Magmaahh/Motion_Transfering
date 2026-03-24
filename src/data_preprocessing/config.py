import torch

# Paths configuration
INPUT_PATH = "src/data_preprocessing/input" # Path to the directory containing smplx sequences to process
MODELS_PATH = "src/data_preprocessing/models" # Path to the directory containing models
OUTPUT_PATH = "src/data_preprocessing/output" # Path to to the directory where outputs (meshes and Anny parameters) are stored
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants configuration
COMPARE = True # Whether to compare the resulting meshes with the original smplx ones
COMPARISON_FRAME = 40 # How often to compare the meshes (every N frames)