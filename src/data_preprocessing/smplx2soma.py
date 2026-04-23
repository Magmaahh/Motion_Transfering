import argparse
import time
import numpy as np
import smplx
import torch
from pathlib import Path

from soma.geometry.rig_utils import remove_joint_orient_local
from soma.geometry.transforms import matrix_to_rotvec
from soma.io import add_npz_args, save_soma_npz
from soma.pose_inversion import PoseInversion
from soma.soma import SOMALayer

from utils import *
from config import *
from vis_pyrender import render_comparison_video

def process_sequence(input_file, args, smplx_model):
    data = np.load(input_file, allow_pickle=True)
    num_frames = len(data['pose_body'])
    gender = str(data['gender']).strip().lower()
    
    base_betas = torch.tensor(data['betas'][:10], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    betas = base_betas.expand(num_frames, -1)
    
    root_orient = torch.tensor(data['root_orient'], dtype=torch.float32).to(DEVICE)
    body_pose = torch.tensor(data['pose_body'], dtype=torch.float32).to(DEVICE)
    transl = torch.tensor(data['trans'], dtype=torch.float32).to(DEVICE)
    
    # Split hands
    hand_pose = torch.tensor(data['pose_hand'], dtype=torch.float32).to(DEVICE)
    left_hand_pose = hand_pose[:, :45] if hand_pose is not None else None
    right_hand_pose = hand_pose[:, 45:] if hand_pose is not None else None
    
    # Jaw and split eyes
    jaw_pose = torch.tensor(data['pose_jaw'], dtype=torch.float32).to(DEVICE)
    eye_pose = torch.tensor(data['pose_eye'], dtype=torch.float32).to(DEVICE)
    leye_pose = eye_pose[:, :3]
    reye_pose = eye_pose[:, 3:]
    
    print(f"Loaded {num_frames} frames")

    soma = SOMALayer(
        ASSETS_PATH,
        identity_model_type="smplx",
        device=DEVICE,
        mode="warp",
    )

    inv = PoseInversion(soma, low_lod=args.low_lod)
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

    # Start inversion
    print("Running inversion...")

    t0 = time.perf_counter()

    all_rot = []
    all_trans = []
    all_err = []

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        current_bs = end - start

        with torch.no_grad():
            expression = torch.zeros(
                (current_bs, smplx_model.num_expression_coeffs), 
                dtype=torch.float32, 
                device=DEVICE
            )

            smplx_out = smplx_model(
                betas=betas[start:end],
                expression=expression,
                global_orient=root_orient[start:end],
                body_pose=body_pose[start:end],
                transl=transl[start:end],
                left_hand_pose=left_hand_pose[start:end],
                right_hand_pose=right_hand_pose[start:end],
                jaw_pose=jaw_pose[start:end],
                leye_pose=leye_pose[start:end],
                reye_pose=reye_pose[start:end],
                return_verts=True
            )

        result = inv.fit(
            smplx_out.vertices,
            body_iters=args.body_iters,
            finger_iters=args.finger_iters,
            full_iters=args.full_iters,
            autograd_iters=args.autograd_iters,
            autograd_lr=args.autograd_lr,
        )

        all_rot.append(result["rotations"].cpu())
        all_trans.append(result["root_translation"].cpu())
        all_err.append(result["per_vertex_error"].cpu())

    rotations = torch.cat(all_rot)
    root_transl = torch.cat(all_trans)
    err = torch.cat(all_err)

    dt = time.perf_counter() - t0

    results = {
        "mean_pve": err.mean().item() * 1000,
        "max_pve": err.max().item() * 1000,
    }

    print(f"Time: {dt:.2f}s ({num_frames/dt:.1f} fps)")
    print(f"Mean error: {results['mean_pve']:.2f} mm")
    print(f"Max error: {results['max_pve']:.2f} mm")

    if args.save:
        # Convert to SOMA format and save
        _soma = inv.soma

        rotations = rotations.to(_soma._t_pose_orient.device)

        rel_rot = remove_joint_orient_local(
            rotations,
            _soma._t_pose_orient,
            _soma._t_pose_orient_parent_T,
        )

        poses_rotvec = matrix_to_rotvec(
            rel_rot.reshape(-1, 3, 3)
        ).reshape(rotations.shape[0], rotations.shape[1], 3)

        output_path = os.path.join(SOMA_OUTPUT_PATH, input_file.stem + "_soma.npz")
        save_soma_npz(
            output_path,
            poses_rotvec,
            root_transl,
            joint_names=list(_soma.rig_data["joint_names"]),
            identity_model_type="smplx",
            identity_coeffs=betas[:1],
            joint_orient=_soma._t_pose_orient,
            unit=args.output_unit,
            keep_root=args.keep_root,
        )

    if args.no_render:
        return results

    # Render: re-run SMPLX forward + SOMA reconstruct in chunks
    _soma = inv.soma
    bs = _soma.batched_skinning
    bind_transforms = _soma._cached_bind_transforms_world
    rest_shape = _soma._cached_rest_shape

    smplx_verts_all = []
    soma_verts_all = []

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        current_bs = end - start

        with torch.no_grad():
            expression = torch.zeros(
                (current_bs, smplx_model.num_expression_coeffs), 
                dtype=torch.float32, 
                device=DEVICE
            )

            smplx_out = smplx_model(
                betas=betas[start:end],
                expression=expression,
                global_orient=root_orient[start:end],
                body_pose=body_pose[start:end],
                transl=transl[start:end],
                left_hand_pose=left_hand_pose[start:end],
                right_hand_pose=right_hand_pose[start:end],
                jaw_pose=jaw_pose[start:end],
                leye_pose=leye_pose[start:end],
                reye_pose=reye_pose[start:end],
                return_verts=True
            )

            smplx_verts_all.append(smplx_out.vertices.cpu().numpy())

            chunk_bind = bind_transforms.expand(end - start, -1, -1, -1)
            chunk_rest = rest_shape.expand(end - start, -1, -1)
            bs.rebind(chunk_bind, chunk_rest)
            with torch.no_grad():
                sv, _ = bs.pose(
                    rotations[start:end].to(DEVICE),
                    root_transl[start:end].to(DEVICE),
                    absolute_pose=True,
                    return_transforms=True,
                )
            soma_verts_all.append(sv.detach().cpu().numpy())
    
    print("\nRendering comparison video...")
    render_comparison_video(
        os.path.join("comparison_videos", input_file.stem + "_comparison.mp4"),
        np.concatenate(smplx_verts_all, axis=0),
        smplx_model.faces,
        np.concatenate(soma_verts_all, axis=0),
        _soma.faces.cpu().numpy(),
        label_source="SMPLX",
        cam_dist_scale=3.0,
    )

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPLX to SOMA pose converter.")
    parser.add_argument("--single-input", type=str, help="Path to the single input SMPLX .npz file to process.")
    parser.add_argument("--body-iters", type=int, default=2, help="Analytical body iterations (default: 2).")
    parser.add_argument("--finger-iters", type=int, default=0, help="Analytical finger iterations (default: 0).")
    parser.add_argument("--full-iters", type=int, default=1, help="Analytical full iterations (default: 1).")
    parser.add_argument("--batch-size", type=int, default=None, help="Process frames in chunks of this size (default: all at once).",)
    parser.add_argument("--autograd-iters", type=int, default=0, help="Autograd FK refinement iterations after analytical solve (default: 0 = off).")
    parser.add_argument("--autograd-lr", type=float, default=5e-3, help="Learning rate for autograd FK (default: 5e-3).",)
    parser.add_argument("--no-render", action="store_true", help="Skip rendering the comparison video.")
    parser.add_argument("--save", action="store_true", help="Save the converted SOMA pose as .npz.")
    parser.add_argument("--low-lod", action="store_true", help="Use low LOD SOMA model for faster processing (not recommended for final results).")
    add_npz_args(parser)
    args = parser.parse_args()

    smplx_model = smplx.create(
        model_type="smplx",
        model_path=MODELS_PATH,
        #gender = gender, # not used since SOMA can only be used with neutral data
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(DEVICE)

    input_path = Path(SOMA_INPUT_PATH)
    npz_files = sorted(input_path.rglob("*.npz"))

    mean_err = []
    max_err = []
    num_failed = 0

    if args.single_input:
        print(f"-- SINGLE INPUT MODE: {args.single_input} --")
        single_file = Path(args.single_input)
        if single_file.is_file() and single_file.suffix == ".npz":
            npz_files = [single_file]
        else:
            print(f"Invalid single input file: {args.single_input}")
            exit(1)
    else:
        print(f"-- BATCH MODE: Processing all sequences in {INPUT_PATH} --")
        
    for file in npz_files:
        print(f"Processing {file}...")

        try:
            results = process_sequence(file, args, smplx_model)

            os.makedirs(SOMA_OUTPUT_PATH, exist_ok=True)
            log_results(file, results, SOMA_OUTPUT_PATH)
            mean_err.append(results["mean_pve"])
            max_err.append(results["max_pve"])
        except Exception as e:
            print(f"Failed to process {file}: {e}")
            num_failed += 1
    
    print("\n=== Summary ===")
    print(f"Processed {len(npz_files)} files with {num_failed} failures.")
    if mean_err:
        print(f"Average Mean PVE: {np.mean(mean_err):.2f} mm")
        print(f"Average Max PVE: {np.mean(max_err):.2f} mm")