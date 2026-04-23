import os
import numpy as np
import trimesh
import trimesh
import numpy as np
import csv
import torch

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
    shift = np.array([offset * 3.5, 0, 0])  # add some spacing

    # Move second mesh to the right
    mesh2.apply_translation(shift)

    scene.add_geometry(mesh1)
    scene.add_geometry(mesh2)

    # Add a third mesh where vertices are assigned a color based on their distance to the target mesh (for better visual comparison)
    distance_colors = np.zeros((verts1.shape[0], 4), dtype=np.uint8)
    max_dist = 0
    max_dist_idx = []
    min_dist = float('inf')
    min_dist_idx = []

    for i, v in enumerate(verts1):
        dist = np.linalg.norm(v - verts2[i])
        if dist >= max_dist:
            max_dist = dist
            max_dist_idx.append(i)
        if dist <= min_dist:
            min_dist = dist
            min_dist_idx.append(i)
        if dist < 0.005:
            distance_colors[i] = [50, 200, 50, 255]  # green for close vertices
        elif dist < 0.01:
            distance_colors[i] = [200, 200, 50, 255]  # yellow for moderately close vertices
        elif dist < 0.015:
            distance_colors[i] = [200, 100, 50, 255]  # orange for far vertices
        else:
            distance_colors[i] = [200, 50, 50, 255]  # red for very far vertices
    
    # Highlight the vertex with the maximum distance in blue
    for idx in max_dist_idx:
        distance_colors[idx] = [200, 50, 200, 255]
    # Highlight the vertex with the minimum distance in purple
    for idx in min_dist_idx:
        distance_colors[idx] = [50, 50, 200, 255]

    distance_mesh = trimesh.Trimesh(vertices=verts1, faces=faces1, vertex_colors=distance_colors)
    distance_mesh.apply_translation(shift * 0.5)
    scene.add_geometry(distance_mesh)

    print("Distances between meshes: (in millimeters)")        
    print(f"Max vertex distance between meshes: {max_dist*1000:.4f}")
    print(f"Min vertex distance between meshes: {min_dist*1000:.4f}")
    print(f"Average vertex distance between meshes: {np.mean(np.linalg.norm(verts1 - verts2, axis=1))*1000:.4f}")

    scene.show()
    
# Plays an animation using given mesh data across frames
def play_animation(full_vertices, faces):
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
    phenotypes_np = {
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
        **phenotypes_np
    )
    print(f"Parameters saved to: {sequence_output_path}")

# Logs training results
def log_results(sequence_name, results, file_path):
    # Log and save training results
    log_fields = [
        "sequence name", "mean pve (mm)", "max pve (mm)"
    ]

    log_path = os.path.join(file_path, "results_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writerow({
            "sequence name": sequence_name,
            "mean pve (mm)": results["mean_pve"],
            "max pve (mm)": results["max_pve"]
        }) 