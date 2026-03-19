import torch
import numpy as np
import smplx
import os
import trimesh

# Paths configuration
INPUT_PATH = "input"
MODELS_PATH = "models"
OUTPUT_PATH = "output"

# Loads the list of .npz files from the input directory
def load_data():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input path not found: {INPUT_PATH}")
    
    # Get .npz files
    files = [f for f in os.listdir(INPUT_PATH) if f.endswith('.npz')]
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {INPUT_PATH}")
    
    return files

# Visualizes the given mesh file
def visualize_mesh(mesh):
    mesh = trimesh.load(mesh)
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    mesh.show()

# Generates a mesh from the SMPL-X model using the provided data content and saves it as an OBJ file
def generate_smplx_mesh(sample, file):
    gender = str(sample['gender']).strip().lower()
    
    # Initialize the SMPL-X model
    model = smplx.create(
        model_path=MODELS_PATH, 
        model_type='smplx',
        gender=gender,
        use_pca=False,
        batch_size=1
    )

    # Prepare the parameters (betas, body_pose, root_orient) for the forward pass
    # Note: Only the first frame (index 0) is taken for body_pose and root_orient
    num_betas = model.num_betas
    betas = torch.tensor(sample['betas'][:num_betas], dtype=torch.float32).unsqueeze(0)
    body_pose = torch.tensor(sample['pose_body'][0:1], dtype=torch.float32)
    root_orient = torch.tensor(sample['root_orient'][0:1], dtype=torch.float32)

    # Forward Pass
    output = model(
        betas=betas,
        body_pose=body_pose,
        root_orient=root_orient,
        return_verts=True
    )

    # Export the mesh to OBJ format
    vertices = output.vertices[0].detach().cpu().numpy()
    faces = model.faces

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # if file not already there
    file_name = file.replace('.npz', '.obj')
    if not os.path.exists(os.path.join(OUTPUT_PATH, file_name)):
        save_path = os.path.join(OUTPUT_PATH, file_name)
        
        with open(save_path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"--- Success! Mesh saved to {save_path}")
        visualize_mesh(save_path)
    else:
        print(f"--- Warning: {file_name} already exists. Skipping save.")

# Main execution block
if __name__ == "__main__":
    # Load the list of .npz files to process
    filelist = load_data()

    # Process each file in the list
    for file in filelist:
        print(f"Processing: {file}")
        
        # Load the .npz file content
        file_path = os.path.join(INPUT_PATH, file)
        try:
            sample = np.load(file_path, allow_pickle=True)
            generate_smplx_mesh(sample, file)
        except Exception as e:
            print(f"Error processing {file}: {e}")