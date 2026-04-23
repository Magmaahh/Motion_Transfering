import argparse
import os
import time
import numpy as np
import smplx
import torch
import torch.nn.functional as F
from pathlib import Path

from soma.geometry.rig_utils import remove_joint_orient_local
from soma.geometry.transforms import matrix_to_rotvec
from soma.io import add_npz_args, save_soma_npz
from soma.pose_inversion import PoseInversion
from soma.soma import SOMALayer

from utils import *
from config import *

def visually_compare_params(soma_smplx, soma_anny, betas, anny_params, rotations, root_transl):
    num_frames = rotations.shape[0]
    test_frames = [0, num_frames // 2, num_frames - 1]

    print(f"\n--- Starting Visual Comparison of {len(test_frames)} frames ---")
    
    for idx in test_frames:
        with torch.no_grad():
            curr_betas = betas[idx:idx+1].to(DEVICE)
            soma_smplx.prepare_identity(curr_betas)

            soma_anny.prepare_identity(anny_params["phenotypes"], anny_params["local_changes"])

            curr_rot = rotations[idx:idx+1].to(DEVICE)
            curr_trans = root_transl[idx:idx+1].to(DEVICE)

            out_smplx = soma_smplx.pose(
                curr_rot, 
                transl=curr_trans, 
                pose2rot=False, 
                absolute_pose=False 
            )
            out_anny = soma_anny.pose(
                curr_rot, 
                transl=curr_trans, 
                pose2rot=False, 
                absolute_pose=False 
            )

        print(f"Visualizing Frame {idx}/{num_frames}...")
        compare_meshes(
            verts1=out_smplx["vertices"][0].cpu().numpy(),
            verts2=out_anny["vertices"][0].cpu().numpy(),
            faces1=soma_smplx.faces.cpu().numpy(),
            faces2=soma_anny.faces.cpu().numpy(),
        )

def optimize_anny_parameters(soma_anny, target_verts, steps=500, lr=1e-2):
    anny_wrapper = soma_anny.identity_model.identity_model
    phenotype_labels = anny_wrapper.phenotype_labels
    local_labels = anny_wrapper.local_change_labels

    n_p = len(phenotype_labels)
    n_l = len(local_labels)

    phenotypes = torch.full((1, n_p), 0.5, dtype=torch.float32, device=DEVICE, requires_grad=True)
    local_changes = torch.zeros((1, n_l), dtype=torch.float32, device=DEVICE, requires_grad=True)

    optimizer = torch.optim.Adam([phenotypes, local_changes], lr=lr)

    patience = 50
    best_loss = 1e10
    best_p = None
    best_l = None

    target_verts = target_verts.to(DEVICE)

    # Optimization loop
    for step in range(steps):
        optimizer.zero_grad()

        # Clamp phenotype range [0, 1]
        phenotypes_clamped = phenotypes.clamp(0.0, 1.0)

        # Clamp local changes range [-3, 3]
        local_changes_clamped = local_changes.clamp(-3.0, 3.0)

        # Build dict inputs
        p_dict = {label: phenotypes_clamped[:, i] for i, label in enumerate(phenotype_labels)}
        l_dict = {label: local_changes_clamped[:, i] for i, label in enumerate(local_labels)}

        # Generate ANNY identity in SOMA space
        soma_anny.prepare_identity(p_dict, l_dict)
        pred_verts = soma_anny._cached_rest_shape

        loss_vertices = F.l1_loss(pred_verts, target_verts)

        # regularization
        loss_local = 0.01 * torch.mean(local_changes ** 2)

        # keep phenotypes near center unless needed
        loss_prior = 0.001 * torch.mean((phenotypes_clamped - 0.5) ** 2)
        loss = loss_vertices + loss_local + loss_prior

        loss.backward()
        optimizer.step()
    
        if loss.item() < best_loss:
            patience = 50  # reset patience
            best_loss = loss.item()
            best_p = phenotypes_clamped.detach().clone()
            best_l = local_changes.detach().clone()
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at step {step} with best loss {best_loss:.6f}")
                break

        if step % 50 == 0:
            print(f"Step {step}: loss={loss.item():.6f}")

    print(f"Best loss: {best_loss:.6f}")

    # rebuild BEST mesh
    best_p_dict = {
        label: best_p[:, i]
        for i, label in enumerate(phenotype_labels)
    }

    best_l_dict = {
        label: best_l[:, i]
        for i, label in enumerate(local_labels)
    }

    with torch.no_grad():
        soma_anny.prepare_identity(best_p_dict, best_l_dict)
        best_verts = soma_anny._cached_rest_shape

    compare_meshes(
        verts1=best_verts[0].cpu().numpy(),
        verts2=target_verts[0].cpu().numpy(),
        faces1=soma_anny.faces.cpu().numpy(),
        faces2=soma_anny.faces.cpu().numpy(),
    )
    
    phenotype_dict = {
        label: best_p[:, i]
        for i, label in enumerate(phenotype_labels)
    }

    local_dict = {
        label: best_l[:, i]
        for i, label in enumerate(local_labels)
    }

    return {
        "phenotypes": phenotype_dict,
        "local_changes": local_dict,
    }

def process_sequence_soma(input_file, args, smplx_model):
    data = np.load(input_file, allow_pickle=True)
    num_frames = len(data['pose_body'])
    
    base_betas = torch.tensor(data['betas'][:10], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    betas = base_betas.expand(num_frames, -1)
    
    root_orient = torch.tensor(data['root_orient'], dtype=torch.float32).to(DEVICE)
    body_pose = torch.tensor(data['pose_body'], dtype=torch.float32).to(DEVICE)
    transl = torch.tensor(data['trans'], dtype=torch.float32).to(DEVICE)
    
    # Split hands/jaw/eyes
    hand_pose = torch.tensor(data['pose_hand'], dtype=torch.float32).to(DEVICE)
    left_hand_pose = hand_pose[:, :45] if hand_pose is not None else None
    right_hand_pose = hand_pose[:, 45:] if hand_pose is not None else None
    
    jaw_pose = torch.tensor(data['pose_jaw'], dtype=torch.float32).to(DEVICE)
    eye_pose = torch.tensor(data['pose_eye'], dtype=torch.float32).to(DEVICE)
    leye_pose = eye_pose[:, :3]
    reye_pose = eye_pose[:, 3:]
    
    print(f"\nProcessing {num_frames} frames from {input_file.name}")

    soma_smplx = SOMALayer(
        ASSETS_PATH,
        identity_model_type="smplx",
        device=DEVICE,
        mode="warp",
    )

    inv = PoseInversion(soma_smplx, low_lod=args.low_lod)
    inv.prepare_identity(base_betas)

    # Warmup
    with torch.no_grad():
        warmup_out = smplx_model(
            betas=betas[:1],
            expression=torch.zeros((1, smplx_model.num_expression_coeffs), device=DEVICE),
            global_orient=root_orient[:1],
            body_pose=body_pose[:1],
            transl=transl[:1],
            left_hand_pose=left_hand_pose[:1],
            right_hand_pose=right_hand_pose[:1],
            jaw_pose=jaw_pose[:1],
            leye_pose=leye_pose[:1],
            reye_pose=reye_pose[:1],
            return_verts=True
        )
    
    inv.fit(
        warmup_out.vertices,
        body_iters=args.body_iters,
        finger_iters=args.finger_iters,
        full_iters=args.full_iters,
        autograd_iters=args.autograd_iters,
        autograd_lr=args.autograd_lr,
    )

    batch_size = args.batch_size or num_frames
    print("--- Running Pose Inversion ---")
    t0 = time.perf_counter()

    all_rot, all_trans, all_err = [], [], []

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        current_bs = end - start

        with torch.no_grad():
            expression = torch.zeros((current_bs, smplx_model.num_expression_coeffs), dtype=torch.float32, device=DEVICE)
            smplx_out = smplx_model(
                betas=betas[start:end], expression=expression, global_orient=root_orient[start:end],
                body_pose=body_pose[start:end], transl=transl[start:end],
                left_hand_pose=left_hand_pose[start:end], right_hand_pose=right_hand_pose[start:end],
                jaw_pose=jaw_pose[start:end], leye_pose=leye_pose[start:end], reye_pose=reye_pose[start:end],
                return_verts=True
            )

        result = inv.fit(
            smplx_out.vertices,
            body_iters=args.body_iters, finger_iters=args.finger_iters, full_iters=args.full_iters,
            autograd_iters=args.autograd_iters, autograd_lr=args.autograd_lr,
        )

        all_rot.append(result["rotations"].cpu())
        all_trans.append(result["root_translation"].cpu())
        all_err.append(result["per_vertex_error"].cpu())

    rotations = torch.cat(all_rot)
    root_transl = torch.cat(all_trans)
    err = torch.cat(all_err)
    
    print(f"Pose Inversion Time: {time.perf_counter() - t0:.2f}s")
    print(f"Mean PVE: {err.mean().item() * 1000:.2f} mm, Max PVE: {err.max().item() * 1000:.2f} mm")

    if args.optimize_anny:
        soma_anny = SOMALayer(
            ASSETS_PATH,
            identity_model_type="anny",
            device=DEVICE,
            mode="warp",
            low_lod=args.low_lod,
        ).to(DEVICE)
        
        anny_params = optimize_anny_parameters(soma_anny, inv.soma._cached_rest_shape.detach())
        #visually_compare_params(soma_smplx, soma_anny, betas, anny_params, rotations, root_transl)

    if args.save:
        base_name = input_file.stem
        output_dir = os.path.join(SOMA_OUTPUT_PATH, base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save SOMA Pose
        _soma = inv.soma
        rotations = rotations.to(_soma._t_pose_orient.device)
        rel_rot = remove_joint_orient_local(rotations, _soma._t_pose_orient, _soma._t_pose_orient_parent_T)
        poses_rotvec = matrix_to_rotvec(rel_rot.reshape(-1, 3, 3)).reshape(rotations.shape[0], rotations.shape[1], 3)

        pose_output_path = os.path.join(output_dir, f"{base_name}_soma_pose.npz")
        save_soma_npz(
            pose_output_path,
            poses_rotvec,
            root_transl,
            joint_names=list(_soma.rig_data["joint_names"]),
            identity_model_type="smplx",
            identity_coeffs=betas[:1],
            joint_orient=_soma._t_pose_orient,
            unit=args.output_unit,
            keep_root=args.keep_root,
        )
        print(f"Saved SOMA pose parameters")

        # 2. Save ANNY Identity if optimized
        if args.optimize_anny:
            print("\nSaving regressed ANNY parameters...")

            phenotypes_np = {
                k: v.cpu().numpy()
                for k, v in anny_params["phenotypes"].items()
            }
            local_changes_np = {
                k: v.cpu().numpy()
                for k, v in anny_params["local_changes"].items()
            } if anny_params["local_changes"] is not None else None

            np.savez(
                os.path.join(output_dir, f"{base_name}_anny_identity.npz"),
                local_changes=local_changes_np,
                **phenotypes_np
            )
            print(f"Saved ANNY shape parameters")

    results = {"mean_pve": err.mean().item() * 1000, "max_pve": err.max().item() * 1000}

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPLX to SOMA Pose and ANNY Shape Parameters Extractor.")
    parser.add_argument("--single-input", type=str, help="Path to the single input SMPLX .npz file to process.")
    parser.add_argument("--body-iters", type=int, default=2)
    parser.add_argument("--finger-iters", type=int, default=0)
    parser.add_argument("--full-iters", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--autograd-iters", type=int, default=0)
    parser.add_argument("--autograd-lr", type=float, default=5e-3)
    parser.add_argument("--save", action="store_true", default=True, help="Save the converted outputs.")
    parser.add_argument("--low-lod", action="store_true", help="Use low LOD SOMA model.")
    parser.add_argument("--optimize-anny", action="store_true", help="Run ANNY parameter optimization.")
    add_npz_args(parser)
    args = parser.parse_args()

    smplx_model = smplx.create(
        model_type="smplx",
        model_path=MODELS_PATH,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(DEVICE)

    input_path = Path(SOMA_INPUT_PATH)
    npz_files = sorted(input_path.rglob("*.npz"))

    if args.single_input:
        npz_files = [Path(args.single_input)]

    for file in npz_files:
        try:
            process_sequence_soma(file, args, smplx_model)
        except Exception as e:
            print(f"Failed to process {file}: {e}")