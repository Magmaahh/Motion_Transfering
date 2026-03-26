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