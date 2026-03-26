import os
import numpy as np
import torch
import trimesh

# Get the list of .npz files in the given directory
def get_data(input_path):
    data = []
    for file in os.listdir(input_path):
        if file.endswith('.npz'):
            data.append(file)

    return data

# Get the value of a specific key at the given frame of the given sample
def get_frame_tensor(sample, key, frame_idx, device):
    if key in sample and len(sample[key]) > frame_idx:
        return torch.tensor(sample[key][frame_idx:frame_idx+1], dtype=torch.float32).to(device)
    
    return None

# Save given mesh data to the specified path
def save_mesh(verts, faces, path):
    with open(path, 'w') as f:
        for v in verts: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces: f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

# Show a mesh using given mesh data
def show_mesh(verts, faces):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.show()

# Compare two meshes visually
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

# Play an animation using given mesh data across frames
def play_animation(full_vertices, faces, fps=30):
    # Create the base mesh using the first frame
    mesh = trimesh.Trimesh(vertices=full_vertices[0], faces=faces)
    scene = trimesh.Scene(mesh)

    # Track the current frame state
    state = {'frame': 0}

    def update_scene(scene):
        # Increment frame
        state['frame'] = (state['frame'] + 1) % len(full_vertices)
        
        # Update the vertices directly in the GPU buffer
        mesh.vertices = full_vertices[state['frame']]
        
    # Open the viewer with the update callback
    scene.show(callback=update_scene)