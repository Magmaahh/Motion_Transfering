import torch
import numpy as np

def sample_motion_diverse_frames(full_vertices, target_ratio=0.10, threshold_factor=0.02):
    """
    Args:
        full_vertices: list of [V,3] tensors, length = T frames.
        target_ratio: desired fraction of frames (~0.1 means keep 10%).
        threshold_factor: scale relative to body size → adaptive threshold
    """
    T = len(full_vertices)
    keep = [0]  # always first (first motion frame)

    # Estimate body scale to normalize threshold
    body_height = full_vertices[0][:,2].max() - full_vertices[0][:,2].min()
    displacement_threshold = threshold_factor * body_height

    last_kept = full_vertices[0]

    for i in range(1, T):
        disp = torch.norm(full_vertices[i] - last_kept, dim=-1).mean()

        if disp > displacement_threshold:
            keep.append(i)
            last_kept = full_vertices[i]

            # stop if we reached target count
            if len(keep) >= int(T * target_ratio):
                break
    
    # If not enough frames were selected, fallback to uniform sampling
    if len(keep) < int(T * target_ratio):
        keep = list(range(0, T, max(1, T // int(T * target_ratio))))

    return keep

def optimize_local_changes(regressor, pose, phenotypes, local_changes, vertices_target, v_hat):
    eps = regressor.eps
    batch_size = pose.shape[0]
    idx = regressor.idx
    keys = list(local_changes.keys())
    D = len(keys)

    # Prepare duplicated batches for FD
    pose_all = pose.unsqueeze(1).repeat(1, D+1, 1, 1, 1).flatten(0,1)
    pheno_all = {k: v.unsqueeze(1).repeat(1, D+1).flatten(0,1) for k, v in phenotypes.items()}
    lc_all = {k: v.unsqueeze(1).repeat(1, D+1).flatten(0,1) for k, v in local_changes.items()}

    # Add eps
    for i, k in enumerate(keys):
        inds = [i+1 + j*(D+1) for j in range(batch_size)]
        lc_all[k][inds] += eps

    # Forward
    verts = regressor.model(
        pose_parameters=pose_all,
        phenotype_kwargs=pheno_all,
        local_changes_kwargs=lc_all,
        pose_parameterization='root_relative_world'
    )["vertices"][:, regressor.unique_ids]

    verts = verts.reshape(batch_size, D+1, -1, 3)
    err = (verts[:, 1:] - verts[:, [0]])  # [B, D, V', 3]
    J = err[:, :, idx].reshape(batch_size, D, -1).transpose(1,2) / eps

    # Compute target residual
    b = (vertices_target[:, idx] - v_hat[:, idx]).reshape(batch_size, -1)

    # Solve least squares per-sample
    delta = torch.linalg.lstsq(J, b).solution  # [B, D]

    # Update local changes
    for i, k in enumerate(keys):
        local_changes[k] += torch.clamp(delta[:, i], -0.1, 0.1)

    return local_changes