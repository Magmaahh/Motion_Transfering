import anny
import smplx
from parameters_regressor import ParametersRegressor

from utils import *

# Samples a subset of frames from the full sequence based on motion diversity (displacement threshold)
def sample_motion_diverse_frames(full_vertices, target_ratio=0.10, threshold_factor=0.02):
    """
    Samples a subset of frames from the full sequence based on motion diversity, using a displacement threshold.

    Args:
        full_vertices: list of [V,3] tensors, with length T equal to the number of frames in the sequence.
        target_ratio: desired fraction of frames (~0.1 means 10% of frames are selected).
        threshold_factor: displacement threshold as a fraction of body height (e.g., 0.02 means 2% of body height).
    Returns:
        A list of frame indices selected based on motion diversity.
    """
    T = len(full_vertices) 

    # Estimate body scale to normalize threshold
    body_height = full_vertices[0][:,2].max() - full_vertices[0][:,2].min()
    displacement_threshold = threshold_factor * body_height

    # Start with the first frame and iteratively add frames that exceed the displacement threshold compared to the last selected frame
    selected = [0]
    last_selected = full_vertices[0]

    for i in range(1, T):
        disp = torch.norm(full_vertices[i] - last_selected, dim=-1).mean()

        if disp > displacement_threshold:
            selected.append(i)
            last_selected = full_vertices[i]

            # Stop when reaching the target ratio of selected frames
            if len(selected) >= 10: # int(T * target_ratio) if enough memory available
                break
    
    # If not enough frames were selected, fallback to uniform sampling
    if len(selected) < 10: # int(T * target_ratio) if enough memory available
        selected = list(range(0, T, max(1, T // int(T * target_ratio))))

    return selected

# Processes an AMASS animation sequence (in SMPL-X format) and translates it into ANNY model parameters
def process_sequence(sequence):
    """
    Processes an AMASS (SMPL-X) animation sequence and translates it into 
    ANNY model parameters (pose, local changes and phenotypes).

    Args:
        sequence: dict containing the SMPL-X parameters for the entire animation sequence.
    Returns:
        A dict containing the regressed ANNY parameters for the entire sequence
        (phenotypes, local changes and pose parameters).
    """
    torch.set_default_dtype(torch.float32)

    # Extract metadata for model initialization and loop boundaries
    gender = str(sequence['gender']).strip().lower()
    num_frames = len(sequence['pose_body'])

    print(f"\n--------------------------------------------------------")
    print(f"\nProcessing sequence: {sequence} with length {num_frames} frames...")

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
    regressor = ParametersRegressor(model=model, verbose=VERBOSE)

    # Extract the SMPL-X shape parameters (constant for the entire sequence)
    betas = torch.tensor(sequence['betas'][:smplx_model.num_betas], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    full_vertices = []
    vertices_for_pheno = []

    # Iterate through each frame in the sequence to extract the corresponding vertices for shape and pose parameters regression
    for f in range(num_frames):
        root_orient = get_frame_tensor(sequence, 'root_orient', f, DEVICE)
        body_pose = get_frame_tensor(sequence, 'pose_body', f, DEVICE)
        transl = get_frame_tensor(sequence, 'trans', f, DEVICE)
        hand_pose = get_frame_tensor(sequence, 'pose_hand', f, DEVICE)

        output = smplx_model(
            betas=betas,
            global_orient=root_orient,
            body_pose=body_pose,
            transl=transl,
            left_hand_pose=hand_pose[:, :45] if hand_pose is not None else None,
            right_hand_pose=hand_pose[:, 45:] if hand_pose is not None else None,
            jaw_pose=get_frame_tensor(sequence, 'pose_jaw', f, DEVICE),
            eye_pose=get_frame_tensor(sequence, 'pose_eye', f, DEVICE),
            return_verts=True
        )

        full_vertices.append(output.vertices[0].detach())

    # Select a subset of frames that are diverse in motion for phenotype regression
    keep = sample_motion_diverse_frames(full_vertices, TARGET_RATIO, DISPLACEMENT_THRESHOLD)

    print(f"\nSelected frames for phenotype regression: {keep}")
    
    # Include the T-pose in the regression set to ensure the model captures the base shape correctly
    tpose_verts = smplx_model(betas=betas, return_verts=True).vertices[0].detach()
    vertices_for_pheno.append(tpose_verts)

    for idx in keep: # add the selected frames for phenotype regression
        vertices_for_pheno.append(full_vertices[idx])

    # Stack into batch [B, V, 3] for regression
    vertices_for_pheno = torch.stack(vertices_for_pheno, dim=0).to(DEVICE)

    # Regress ANNY shape parameters (shared across all sampled frames)
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

    print("\nRegressed local changes:")
    for k, v in local_changes.items():
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

            # Compare the meshes visually (first (SMPL-X) in red, second (ANNY) in blue)
            print(f"\nComparing meshes at frame {f}... (SMPL-X in red, ANNY in blue)")
            compare_meshes(
                full_vertices_np[f], # SMPL-X target vertices
                fitted_verts_np, # ANNY fitted vertices
                model.faces, # ANNY faces
                smplx_model.faces # SMPL-X faces
            )

    return {"phenotypes": phenotypes, "local_changes": local_changes, "pose_params": all_pose_params}
    