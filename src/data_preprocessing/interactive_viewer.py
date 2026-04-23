import gradio as gr
import torch
import anny
import roma
import tempfile
import trimesh
import numpy as np

def interactive_compare_meshes(model_type, mesh_data, parameters, reference_model_type, reference_mesh_data, server_name: str = None, server_port: int = None):
    dtype = torch.float32
    
    # Check if anny is involved in this specific comparison
    is_main_anny = model_type.lower() == "anny"
    is_ref_anny = reference_model_type.lower() == "anny"
    has_anny = is_main_anny or is_ref_anny

    with (tempfile.NamedTemporaryFile(suffix=".glb") as temp_file,
          tempfile.NamedTemporaryFile(suffix=".json") as temp_params_file):

        mesh_filename = temp_file.name
        
        model = None
        bones_rotvec = None
        phenotype_kwargs = {}
        local_changes_kwargs = {}

        if has_anny and parameters is not None:
            pose_params_in = parameters.get("pose_params")
            pose_params = pose_params_in.detach().cpu() if isinstance(pose_params_in, torch.Tensor) else pose_params_in

        def to_float(val):
            return float(val.item()) if isinstance(val, torch.Tensor) else float(val)

        def get_vertices_and_faces(m_data, is_anny_flag):
            """Helper to fetch dynamic anny vertices or static mesh vertices"""
            if is_anny_flag and model is not None:
                output = model(
                    pose_parameters=pose_params,
                    phenotype_kwargs=phenotype_kwargs,
                    local_changes_kwargs=local_changes_kwargs,
                    pose_parameterization='root_relative_world'
                )
                verts = output['vertices'][0].detach().cpu().numpy()
                faces = model.faces.cpu().numpy()
            else:
                verts = m_data["verts"].detach().cpu().numpy() if isinstance(m_data["verts"], torch.Tensor) else m_data["verts"]
                faces = m_data["faces"].detach().cpu().numpy() if isinstance(m_data["faces"], torch.Tensor) else m_data["faces"]

            return verts, faces

        def export_mesh(show_distances=False):
            # Conditionally generate or fetch vertices
            main_verts, main_faces = get_vertices_and_faces(mesh_data, is_main_anny)
            ref_verts, ref_faces = get_vertices_and_faces(reference_mesh_data, is_ref_anny)

            scene = trimesh.Scene()
            
            # Distance computation logic
            pve_val, max_val, min_val = 0.0, 0.0, 0.0
            stats_texts = ["N/A", "N/A", "N/A"]

            if show_distances and len(main_verts) == len(ref_verts):
                distances = np.linalg.norm(ref_verts - main_verts, axis=1)
                
                pve_val = float(np.mean(distances)) * 1000
                max_val = float(np.max(distances)) * 1000
                min_val = float(np.min(distances)) * 1000
                stats_texts = [f"{pve_val:.6f}", f"{max_val:.6f}", f"{min_val:.6f}"]

                distance_colors = np.zeros((len(ref_verts), 4), dtype=np.uint8)
                distance_colors[distances <= 0.005] = [50, 200, 50, 255]
                distance_colors[(distances > 0.005) & (distances < 0.01)] = [200, 200, 50, 255]
                distance_colors[(distances >= 0.01) & (distances < 0.015)] = [200, 100, 50, 255]
                distance_colors[distances >= 0.015] = [200, 50, 50, 255]

                distance_colors[distances == max_val] = [200, 50, 200, 255]
                distance_colors[distances == min_val] = [50, 50, 200, 255]

                reference_mesh = trimesh.Trimesh(vertices=ref_verts, faces=ref_faces, vertex_colors=distance_colors)
            else:
                reference_mesh = trimesh.Trimesh(vertices=ref_verts, faces=ref_faces)
                reference_mesh.visual.vertex_colors = [200, 50, 50, 255] # Red for reference mesh when not showing distances
            
            main_mesh = trimesh.Trimesh(vertices=main_verts, faces=main_faces)
            main_mesh.visual.vertex_colors = [50, 50, 200, 255] # Blue for main mesh
            scene.add_geometry(main_mesh)

            offset = reference_mesh.bounds[1][0] - reference_mesh.bounds[0][0]
            shift = np.array([offset * 3.5, 0, 0])

            reference_mesh.apply_translation(shift)
            scene.add_geometry(reference_mesh)

            center = scene.bounds.mean(axis=0)
            scene.apply_translation(-center)
            scale = 1.0 / np.max(scene.extents)
            scene.apply_scale(scale)

            R = roma.euler_to_rotmat('xyz', [-270., 0., 180.], degrees=True)
            scene.apply_transform(roma.Rigid(R, torch.zeros(3)).to_homogeneous().numpy())

            scene.export(mesh_filename)

            return mesh_filename, temp_params_file.name, stats_texts[0], stats_texts[1], stats_texts[2]
        
        def initialize_model(rig="default"):
            nonlocal model, bones_rotvec, phenotype_kwargs, local_changes_kwargs
            
            # Only initialize Anny model if it's needed
            if has_anny:
                topology = "default"
                if reference_model_type.lower() == "smplx" or model_type.lower() == "smplx":
                    topology = "smplx"

                model = anny.create_fullbody_model(rig=rig, topology=topology, local_changes=True)
                model = model.to(dtype=dtype)
                bones_rotvec = torch.zeros((len(model.bone_labels), 3), dtype=dtype)
                
                in_phenotypes = parameters.get("phenotypes", {}) if parameters else {}
                phenotype_kwargs = {
                    key: to_float(in_phenotypes.get(key, 0.5)) for key in model.phenotype_labels
                }
                
                in_local = parameters.get("local_changes") or {}
                local_changes_kwargs = {
                    key: to_float(in_local.get(key, 0.)) for key in model.local_change_labels
                }

                desc_text = "\n".join([
                    f"- Vertices: {len(model.template_vertices)}",
                    f"- Faces: {len(model.faces)}",
                    f"- Bones: {len(model.bone_labels)}",
                    f"- Blendshapes: {model.blendshapes.shape[0]}",
                    f"- Max influencing bones: {model.vertex_bone_weights.shape[1]}"
                ])
                phenotype_choices = model.phenotype_labels
                local_choices = model.local_change_labels
            else:
                desc_text = "Standard mesh comparison."
                phenotype_choices = ["None"]
                local_choices = ["None"]
                phenotype_kwargs = {"None": 0.0}
                local_changes_kwargs = {"None": 0.0}

            description = gr.Markdown(desc_text)
            
            phenotype_dropdown = gr.Dropdown(label="Phenotype", choices=phenotype_choices, value=phenotype_choices[0])
            macrodetail_slider = gr.Slider(label="Value", minimum=0., maximum=1., step=0.05, value=phenotype_kwargs.get(phenotype_choices[0], 0.0))
            
            if len(local_choices) > 0 and local_choices[0] != "None":
                local_change_dropdown = gr.Dropdown(label="Local change", choices=local_choices, value=local_choices[0], interactive=True)
                local_changes_slider = gr.Slider(label="Value", minimum=-3., maximum=3., step=0.05, value=local_changes_kwargs.get(local_choices[0], 0.0), interactive=True)
            else:
                local_change_dropdown = gr.Dropdown(label="Local change", choices=["None"], value="None", interactive=False)
                local_changes_slider = gr.Slider(label="Value", minimum=-3., maximum=3., step=0.05, value=0., interactive=False)
                
            reset_shape_button = gr.Button("Reset shape")
            
            show_distances_checkbox = gr.Checkbox(label="Show vertices distances (in mm)", value=False)
            pve_text = gr.Textbox(label="PVE (average)", value="N/A", interactive=False)
            max_text = gr.Textbox(label="MAX distance", value="N/A", interactive=False)
            min_text = gr.Textbox(label="MIN distance", value="N/A", interactive=False)

            filename, params_filename, init_pve, init_max, init_min = export_mesh(show_distances=False)
            model3d = gr.Model3D(value=filename, height="100vh")
            
            return (description, phenotype_dropdown, macrodetail_slider, local_change_dropdown, 
                    local_changes_slider, reset_shape_button, show_distances_checkbox, 
                    pve_text, max_text, min_text, model3d)

        # Initialize UI Components
        (description, phenotype_dropdown, macrodetail_slider, local_change_dropdown, 
         local_changes_slider, reset_shape_button, show_distances_checkbox, 
         pve_text, max_text, min_text, model3d) = initialize_model()

        with gr.Blocks(title="Model Comparison", css="#control-column { max-width: 60pt; }") as demo:
            with gr.Row():
                with gr.Column("compact", elem_id="control-column"):
                    gr.Markdown(f"## 3D Model Comparison Tool between <span style='color: red;'>{reference_model_type.upper()}</span> and <span style='color: blue;'>{model_type.upper()}</span>", elem_id="title")
                    gr.Markdown("### Model Information")
                    description.render()

                    # Conditionally show Anny controls
                    with gr.Column(visible=has_anny):
                        gr.Markdown("### Shape Controls")
                        phenotype_dropdown.render()
                        macrodetail_slider.render()
                        local_change_dropdown.render()
                        local_changes_slider.render()
                        reset_shape_button.render() 
                    
                    gr.Markdown("### Analytics")
                    show_distances_checkbox.render()
                    pve_text.render()
                    max_text.render()
                    min_text.render()
                    
                    # Split Upload and Download functionality correctly
                    with gr.Row(visible=has_anny):
                        upload_params_button = gr.UploadButton(label="Upload params", file_types=[".npz"])
                        download_params_button = gr.DownloadButton(label="Download params", value=temp_params_file.name)
                
                model3d.render()

            # Event Listeners
            if has_anny:
                def load_uploaded_params(file_obj, current_phenotype, current_local, show_dists):
                    if file_obj is None:
                        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                        
                    data = np.load(file_obj.name, allow_pickle=True)
                    
                    # Parse Phenotypes (saved as **kwargs in your script)
                    for key in model.phenotype_labels:
                        if key in data:
                            phenotype_kwargs[key] = float(data[key].item())
                            
                    # Parse Local Changes (saved as a dict inside a 0D array in your script)
                    if "local_changes" in data and data["local_changes"].shape == ():
                        local_dict = data["local_changes"].item()
                        if isinstance(local_dict, dict):
                            for key in model.local_change_labels:
                                if key in local_dict:
                                    local_changes_kwargs[key] = float(local_dict[key].item())
                    
                    # Ensure sliders update accurately for the currently active dropdowns
                    new_macro_val = phenotype_kwargs.get(current_phenotype, 0.5)
                    new_local_val = local_changes_kwargs.get(current_local, 0.0)
                    
                    # Trigger the mesh generation
                    mesh_f, param_f, pve_v, max_v, min_v = export_mesh(show_dists)
                    
                    return mesh_f, param_f, new_macro_val, new_local_val, pve_v, max_v, min_v

                upload_params_button.upload(
                    load_uploaded_params,
                    inputs=[upload_params_button, phenotype_dropdown, local_change_dropdown, show_distances_checkbox],
                    outputs=[model3d, download_params_button, macrodetail_slider, local_changes_slider, pve_text, max_text, min_text]
                )

                def update_phenotype_label(macrodetail_label):
                    return phenotype_kwargs[macrodetail_label]
                phenotype_dropdown.change(update_phenotype_label, inputs=phenotype_dropdown, outputs=macrodetail_slider)
                
                def update_phenotype_slider(macrodetail_label, value, show_dists):
                    phenotype_kwargs[macrodetail_label] = value
                    return export_mesh(show_dists)
                macrodetail_slider.change(update_phenotype_slider, inputs=[phenotype_dropdown, macrodetail_slider, show_distances_checkbox], outputs=[model3d, download_params_button, pve_text, max_text, min_text])

                def update_local_changes_label(local_changes_label):
                    return local_changes_kwargs.get(local_changes_label, 0.)
                local_change_dropdown.change(update_local_changes_label, inputs=local_change_dropdown, outputs=local_changes_slider)

                def update_local_changes_slider(local_changes_label, value, show_dists):
                    if local_changes_kwargs is not None:
                        local_changes_kwargs[local_changes_label] = value
                    return export_mesh(show_dists)
                local_changes_slider.change(update_local_changes_slider, inputs=[local_change_dropdown, local_changes_slider, show_distances_checkbox], outputs=[model3d, download_params_button, pve_text, max_text, min_text])

                def reset_shape(macrodetail_label, local_change_label, show_dists):
                    in_phenotypes = parameters.get("phenotypes", {}) if parameters else {}
                    phenotype_kwargs.update({
                        key: to_float(in_phenotypes.get(key, 0.5)) for key in model.phenotype_labels
                    })
                    
                    if local_changes_kwargs is not None:
                        in_local = parameters.get("local_changes", {}) if parameters else {}
                        local_changes_kwargs.update({
                            key: to_float(in_local.get(key, 0.)) for key in model.local_change_labels
                        })
                    
                    local_change_output = local_changes_kwargs[local_change_label] if local_changes_kwargs else 0.
                    macro_change_output = phenotype_kwargs[macrodetail_label]
                    
                    mesh_f, param_f, pve_v, max_v, min_v = export_mesh(show_dists)
                    return mesh_f, param_f, macro_change_output, local_change_output, pve_v, max_v, min_v
                
                reset_shape_button.click(
                    reset_shape, 
                    inputs=[phenotype_dropdown, local_change_dropdown, show_distances_checkbox], 
                    outputs=[model3d, download_params_button, macrodetail_slider, local_changes_slider, pve_text, max_text, min_text]
                )

            # Universal Event Listeners
            def toggle_distances(show_dists):
                return export_mesh(show_dists)
            show_distances_checkbox.change(toggle_distances, inputs=[show_distances_checkbox], outputs=[model3d, download_params_button, pve_text, max_text, min_text])

        demo.launch(server_name=server_name, server_port=server_port)