import os
import anny
import smplx
import torch
import numpy as np
from parameters_regressor import ParametersRegressor

from config import *
from utils import *
from functions import *

# Process a single AMASS animation sequence (in SMPL-X format) and translate it into ANNY model parameters
def process_sequence(file_name):
    """
    Processes an AMASS (SMPL-X) animation sequence and translates it into 
    ANNY model parameters (pose, local changes and phenotypes).
    """
    torch.set_default_dtype(torch.float32)

    # Load the AMASS sample data (.npz) containing the SMPL-X parameters to process from the given file
    file_path = os.path.join(INPUT_PATH, file_name)
    sample = np.load(file_path, allow_pickle=True)
    
    # Extract metadata for model initialization and loop boundaries
    gender = str(sample['gender']).strip().lower()
    num_frames = len(sample['pose_body'])

    # Initialize the source SMPL-X model
    smplx_model = smplx.create(
        model_path=MODELS_PATH, 
        model_type='smplx',
        gender=gender, 
        use_pca=False, 
        flat_hand_mean=False, 
        batch_size=1
    ).to(DEVICE)

    # Initialize the target ANNY model (using the SMPL-X topology)
    model = anny.create_fullbody_model(rig="default", topology="smplx", local_changes=True).to(DEVICE).float()
    
    # Instantiate the regressor used to map SMPL-X vertices to ANNY parameters
    regressor = ParametersRegressor(model=model, verbose=True)

    # Extract the smplx shape parameters (constant for the entire sequence)
    betas = torch.tensor(sample['betas'][:smplx_model.num_betas], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    full_vertices = []
    vertices_for_pheno = []

    # Always include the T-pose frame for phenotype regression
    tpose_verts = smplx_model(betas=betas, return_verts=True).vertices[0].detach()
    full_vertices.append(tpose_verts)

    # Iterate through each frame in the sequence to extract the corresponding vertices for phenotype regression and pose tracking
    for f in range(num_frames):
        root_orient = get_frame_tensor(sample, 'root_orient', f, DEVICE)
        body_pose = get_frame_tensor(sample, 'pose_body', f, DEVICE)
        transl = get_frame_tensor(sample, 'trans', f, DEVICE)
        hand_pose = get_frame_tensor(sample, 'pose_hand', f, DEVICE)

        output = smplx_model(
            betas=betas,
            global_orient=root_orient,
            body_pose=body_pose,
            transl=transl,
            left_hand_pose=hand_pose[:, :45] if hand_pose is not None else None,
            right_hand_pose=hand_pose[:, 45:] if hand_pose is not None else None,
            jaw_pose=get_frame_tensor(sample, 'pose_jaw', f, DEVICE),
            eye_pose=get_frame_tensor(sample, 'pose_eye', f, DEVICE),
            return_verts=True
        )

        full_vertices.append(output.vertices[0].detach())

    # Select a subset of frames that are diverse in motion for phenotype regression (including the T-pose)
    keep = sample_motion_diverse_frames(full_vertices, TARGET_RATIO, DISPLACEMENT_THRESHOLD)
    print(f"\nSelected frames for phenotype regression: {keep}")
    vertices_for_pheno.append(tpose_verts) # always include T-pose
    for idx in keep: # add the selected frames for phenotype regression
        vertices_for_pheno.append(full_vertices[idx])

    # Stack into batch [B, V, 3] for regression
    vertices_for_pheno = torch.stack(vertices_for_pheno, dim=0).to(DEVICE)

    # Regress ANNY shape phenotypes (shared across all sampled frames)
    _, phenotypes, local_changes, _ = regressor(
        vertices_target=vertices_for_pheno,
        optimize_phenotypes=True,
        shared_phenotypes=True
    )

    # Convert phenotypes to 1D for frame-wise pose tracking
    phenotypes = {k: v[:1].detach().clone() for k, v in phenotypes.items()}

    # Convert local changes to 1D for frame-wise pose tracking (if they exist, otherwise set to None)
    local_changes = {k: v[:1].detach().clone() for k, v in local_changes.items()} if local_changes is not None else None

    print("\nRegressed ANNY phenotypes:")
    for k, v in phenotypes.items():
        print(f"{k}: {v.item():.4f}")

    # Fit ANNY pose parameters frame-by-frame while keeping phenotypes frozen
    prev_pose = None
    all_pose_params = []

    # Iterate through every frame in the sequence
    print("\nTracking pose across frames...")
    for f in range(num_frames):
        # Regress ANNY pose parameters to match the SMPL-X vertices
        pose_params, _, _, _ = regressor(
            vertices_target=full_vertices[f], # Use the current frame's vertices
            initial_phenotype_kwargs=phenotypes, # Keep phenotypes frozen
            initial_local_changes_kwargs=local_changes, # Keep local changes frozen
            initial_pose_parameters=prev_pose, # Warm-start with the previous frame's pose
            optimize_phenotypes=False
        )

        # Store the current pose to be used in the next frame
        prev_pose = pose_params.detach().float()
        all_pose_params.append(prev_pose[0].cpu().numpy())

        full_vertices_np = [v.cpu().numpy() for v in full_vertices]

        # Visually compare the SMPL-X target vs ANNY prediction (every COMPARISON_FRAME frame, if COMPARE is enabled)
        if COMPARE and f % COMPARE_FRAME == 0:
            fitted_output = model(
                pose_parameters=pose_params,
                phenotype_kwargs=phenotypes,
                local_changes_kwargs=local_changes,
                pose_parameterization='root_relative_world'
            )
            fitted_verts_np = fitted_output['vertices'][0].detach().cpu().numpy()

            # Compare the meshes visually (first (SMPLX) in red, second (ANNY) in blue)
            print(f"\nComparing meshes at frame {f}... (SMPL-X in red, ANNY in blue)")
            compare_meshes(
                full_vertices_np[f], # SMPL-X target vertices
                fitted_verts_np, # ANNY fitted vertices
                model.faces, # ANNY faces
                smplx_model.faces # SMPL-X faces
            )

    # Save regressed ANNY parameters
    print("\nSaving regressed ANNY parameters...")
    anim_name = file_name.replace('.npz', '')
    anim_output_path = os.path.join(OUTPUT_PATH, anim_name)
    os.makedirs(anim_output_path, exist_ok=True)

    all_pose_params = np.stack(all_pose_params, axis=0)
    phenotype_np = {
        k: v.cpu().numpy()
        for k, v in phenotypes.items()
    }
    local_changes_np = {
        k: v.cpu().numpy()
        for k, v in local_changes.items()
    } if local_changes is not None else None

    np.savez(
        os.path.join(anim_output_path, f"{anim_name}_anny_params.npz"),
        poses=all_pose_params,
        local_changes=local_changes_np,
        **phenotype_np
    )
    print(f"Parameters saved to: {anim_output_path}")

# Main execution loop
if __name__ == "__main__":
    # Retrieve all target sequences from the input directory
    data = get_data(INPUT_PATH)
    
    # Process each sequence individually
    for file in data:
        process_sequence(file)