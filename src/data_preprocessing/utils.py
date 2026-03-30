import os
import numpy as np
import trimesh

from config import *

# Gets the list of .npz files in the given folder
def get_sequences(folder_path):
    sequences = []

    print(f"\nRetrieving sequences from folder: {folder_path}...")
    for file in os.listdir(folder_path):
        if file.endswith('.npz'):
            sequences.append(file)
    print(f"Found {len(sequences)} sequences: {sequences}")

    return sequences

# Gets the tensor for a specific frame and key from the given sample
def get_frame_tensor(sample, key, frame_idx, device):
    if key in sample and len(sample[key]) > frame_idx:
        return torch.tensor(sample[key][frame_idx:frame_idx+1], dtype=torch.float32).to(device)
    
    return None

# Saves given mesh data to the specified path
def save_mesh(verts, faces, path):
    with open(path, 'w') as f:
        for v in verts: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces: f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

# Shows a mesh using given mesh data
def show_mesh(verts, faces):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.show()

# Compares two meshes visually
def compare_meshes(verts1, verts2, faces1, faces2):
    scene = trimesh.Scene()
    mesh1 = trimesh.Trimesh(vertices=verts1, faces=faces1)
    mesh1.visual.vertex_colors = [200, 50, 50, 255]  # red
    mesh2 = trimesh.Trimesh(vertices=verts2, faces=faces2)
    mesh2.visual.vertex_colors = [50, 50, 200, 255]  # blue

    # Compute shift based on bounding box size (so they don't overlap)
    offset = mesh1.bounds[1][0] - mesh1.bounds[0][0]  # width in x
    shift = np.array([offset * 1.5, 0, 0])  # add some spacing

    # Move second mesh to the right
    mesh2.apply_translation(shift)

    scene.add_geometry(mesh1)
    scene.add_geometry(mesh2)

    scene.show()

# Plays an animation using given mesh data across frames
def play_animation(full_vertices, faces, fps=30):
    # Create the base mesh using the first frame
    mesh = trimesh.Trimesh(vertices=full_vertices[0], faces=faces)
    scene = trimesh.Scene(mesh)

    # Track the current frame state
    state = {'frame': 0}

    def update_scene(scene):
        state['frame'] = (state['frame'] + 1) % len(full_vertices)
        mesh.vertices = full_vertices[state['frame']]
        
    scene.show(callback=update_scene)

# Saves regressed ANNY parameters
def save_parameters(params, sequence_output_path, sequence_name):
    print("\nSaving regressed ANNY parameters...")
    all_pose_params = np.stack(params["pose_params"], axis=0)
    phenotype_np = {
        k: v.cpu().numpy()
        for k, v in params["phenotypes"].items()
    }
    local_changes_np = {
        k: v.cpu().numpy()
        for k, v in params["local_changes"].items()
    } if params["local_changes"] is not None else None

    np.savez(
        os.path.join(sequence_output_path, f"{sequence_name}_anny_params.npz"),
        poses=all_pose_params,
        local_changes=local_changes_np,
        **phenotype_np
    )
    print(f"Parameters saved to: {sequence_output_path}")