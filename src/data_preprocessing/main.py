import os
import anny
import smplx
import torch
import numpy as np
from anny import ParametersRegressor

from config import *
from utils import *

# Process a single AMASS animation sequence (in SMPL-X format) and translate it into ANNY model parameters
def process_sequence(file_name):
    """
    Processes an AMASS (SMPL-X) animation sequence and translates it into 
    ANNY model parameters (pose and phenotypes).
    """
    torch.set_default_dtype(torch.float32)

    # Load the AMASS sample data (.npz) containing SMPL-X parameters from the given file
    file_path = os.path.join(INPUT_PATH, file_name)
    sample = np.load(file_path, allow_pickle=True)
    
    # Extract metadata for model initialization and loop boundaries
    gender = str(sample['gender']).strip().lower()
    num_frames = len(sample['pose_body'])

    print(f"\n{'='*50}")
    print(f"\nProcessing: {file_name} | {num_frames} frames | {gender}")

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
    model = anny.models.retopology.create_smplx_topology_model().to(DEVICE).float()
    
    # Instantiate the regressor used to map SMPL-X vertices to ANNY parameters
    regressor = ParametersRegressor(model=model, verbose=True)

    # Extract the shape parameters (constant for the entire sequence)
    betas = torch.tensor(
        sample['betas'][:smplx_model.num_betas],
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)
    
    # Set up output directory structure based on the animation name
    anim_name = file_name.replace('.npz', '')
    anim_output_path = os.path.join(OUTPUT_PATH, anim_name)
    os.makedirs(anim_output_path, exist_ok=True)

    # Fit ANNY shape parameters (=phenotypes) to a clean T-pose
    print("\nFitting base phenotypes on T-Pose...")

    # Extract the T-pose vertices from the SMPL-X model
    tpose_output = smplx_model(betas=betas, return_verts=True)
    tpose_verts = tpose_output.vertices.float()

    # Regress the ANNY shape parameters (phenotypes)
    print("Regressing ANNY shape parameters (phenotypes)")
    pose_params, phenotypes, _ = regressor.fit_with_age_anchor_search(
        vertices_target=tpose_verts,
        optimize_phenotypes=True,
        max_n_iters=30
    )
    print("Phenotypes successfully frozen")

    fitted_output = model(
        pose_parameters=pose_params,
        phenotype_kwargs=phenotypes,
        pose_parameterization='root_relative_world'
    )
    fitted_verts = fitted_output['vertices'][0].detach().cpu().numpy()

    compare_meshes(
        tpose_verts[0].detach().cpu().numpy(),
        fitted_verts,
        model.faces,
        smplx_model.faces
    )

    pve = torch.norm(fitted_output['vertices'] - tpose_verts, dim=-1).mean() * 1000
    print(f"Phenotype Regressor PVE on T-pose: {pve.item():.2f} mm")

    # Fit ANNY pose parameters frame-by-frame while keeping phenotypes frozen
    prev_pose = None
    all_pose_params = []
    full_vertices = []

    # Fit ANNY pose parameters frame-by-frame while keeping phenotypes frozen
    prev_pose = None
    all_pose_params = []
    full_vertices = []

    # Iterate through every frame in the sequence
    print("\nTracking pose across frames...")
    for f in range(num_frames):
        # Extract SMPL-X pose data for the current frame
        root_orient = get_frame_tensor(sample, 'root_orient', f, DEVICE)
        body_pose = get_frame_tensor(sample, 'pose_body', f, DEVICE)
        transl = get_frame_tensor(sample, 'trans', f, DEVICE)
        hand_pose = get_frame_tensor(sample, 'pose_hand', f, DEVICE)
        
        # Generate target 3D mesh (vertices) using the SMPL-X model
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
        verts = output.vertices.float()
        verts_np = verts[0].detach().cpu().numpy()
        full_vertices.append(verts_np)

        # Regress the ANNY pose parameters to match the SMPL-X vertices
        pose_params, _, _ = regressor(
            vertices_target=verts,
            initial_phenotype_kwargs=phenotypes, # Keep phenotypes frozen to the T-pose fit
            initial_pose_parameters=prev_pose, # Warm-start with the previous frame's pose
            optimize_phenotypes=False
        )

        # Store the current pose to be used in the next frame
        prev_pose = pose_params.detach().float()
        all_pose_params.append(prev_pose[0].cpu().numpy())

        # Visually compare the SMPL-X target vs ANNY prediction periodically (every COMPARISON_FRAME frame, if COMPARE is enabled)
        if COMPARE and f % COMPARISON_FRAME == 0:
            fitted_output = model(
                pose_parameters=pose_params,
                phenotype_kwargs=phenotypes,
                pose_parameterization='root_relative_world'
            )
            fitted_verts = fitted_output['vertices'][0].detach().cpu().numpy()

            compare_meshes(
                verts_np,
                fitted_verts,
                model.faces,
                smplx_model.faces
            )

    # Save regressed ANNY parameters
    print("\nSaving regressed ANNY parameters...")
    all_pose_params = np.stack(all_pose_params, axis=0)
    
    phenotype_np = {
        k: v.cpu().numpy()
        for k, v in phenotypes.items()
    }
    np.savez(
        os.path.join(anim_output_path, f"{anim_name}_anny_params.npz"),
        poses=all_pose_params,
        **phenotype_np
    )
    print(f"Parameters saved to: {anim_output_path}")

    play_animation(full_vertices, smplx_model.faces)

# Main execution loop
if __name__ == "__main__":
    # Retrieve all target sequences from the input directory
    data = get_data(INPUT_PATH)
    
    # Process each sequence individually
    for file in data:
        process_sequence(file)