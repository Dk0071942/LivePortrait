# coding: utf-8

"""
The entrance of the gradio for animal
"""

import os
import tyro
import subprocess
import gradio as gr
import os.path as osp
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipelineAnimal
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.enhancement_config import EnhancementConfig


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


# set tyro theme
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
if osp.exists(ffmpeg_dir):
    os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
    )
# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
crop_cfg = partial_fields(CropConfig, args.__dict__)
enh_cfg = partial_fields(EnhancementConfig, args.__dict__) # use attribute of args to initial CropConfig

gradio_pipeline_animal: GradioPipelineAnimal = GradioPipelineAnimal(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    enh_cfg=enh_cfg,
    args=args
)

if args.gradio_temp_dir not in (None, ''):
    os.environ["GRADIO_TEMP_DIR"] = args.gradio_temp_dir
    os.makedirs(args.gradio_temp_dir, exist_ok=True)

def gpu_wrapped_execute_video(*args, **kwargs):
    # The main "Animate" button provides 16 positional arguments.
    # The 16th argument (index 15) is the value from flag_enhance_input.
    num_inputs_for_main_button = 16
    # idx_flag_enhance_input_value = 15 # We will now use named args

    if len(args) == num_inputs_for_main_button:
        # Explicitly map args to named variables for clarity and debugging
        ui_source_image_input             = args[0]
        ui_driving_video_input            = args[1]
        ui_driving_video_pickle_input     = args[2]
        ui_flag_do_crop_input             = args[3]
        ui_flag_remap_input               = args[4]
        ui_driving_multiplier             = args[5]
        ui_flag_stitching                 = args[6]
        ui_flag_crop_driving_video_input  = args[7]
        ui_scale                          = args[8]
        ui_vx_ratio                       = args[9]
        ui_vy_ratio                       = args[10]
        ui_scale_crop_driving_video       = args[11]
        ui_vx_ratio_crop_driving_video    = args[12]
        ui_vy_ratio_crop_driving_video    = args[13]
        ui_tab_selection                  = args[14] # Crucial for driving input type
        ui_flag_enhance_input             = args[15]

        print(f"[DEBUG app_animals.py gpu_wrapped_execute_video] ui_tab_selection from args[14]: '{ui_tab_selection}'")
        print(f"[DEBUG app_animals.py gpu_wrapped_execute_video] ui_flag_enhance_input from args[15]: {ui_flag_enhance_input}")

        original_pipeline_enh_cfg_flag_enhance = gradio_pipeline_animal.enh_cfg.flag_enhance

        try:
            gradio_pipeline_animal.enh_cfg.flag_enhance = ui_flag_enhance_input

            print(f"[DEBUG app_animals.py] Before execute_video: UI flag_enhance_input: {ui_flag_enhance_input}")
            print(f"[DEBUG app_animals.py] Before execute_video: pipeline.enh_cfg.flag_enhance: {gradio_pipeline_animal.enh_cfg.flag_enhance}")

            result = gradio_pipeline_animal.execute_video(
                input_source_image_path=ui_source_image_input,
                input_driving_video_path=ui_driving_video_input,
                input_driving_video_pickle_path=ui_driving_video_pickle_input,
                flag_do_crop_input=ui_flag_do_crop_input,
                flag_remap_input=ui_flag_remap_input,
                driving_multiplier=ui_driving_multiplier,
                flag_stitching=ui_flag_stitching,
                flag_crop_driving_video_input=ui_flag_crop_driving_video_input,
                scale=ui_scale,
                vx_ratio=ui_vx_ratio,
                vy_ratio=ui_vy_ratio,
                scale_crop_driving_video=ui_scale_crop_driving_video,
                vx_ratio_crop_driving_video=ui_vx_ratio_crop_driving_video,
                vy_ratio_crop_driving_video=ui_vy_ratio_crop_driving_video,
                tab_selection=ui_tab_selection, # Pass it explicitly
                flag_enhance=ui_flag_enhance_input
            )
        finally:
            gradio_pipeline_animal.enh_cfg.flag_enhance = original_pipeline_enh_cfg_flag_enhance
        return result
    else:
        # This branch handles calls with a different number of arguments (e.g., from examples).
        # For these calls, the enhancement behavior will depend on the pipeline's
        # default handling (likely using the initial CLI-derived self.enh_cfg.flag_enhance
        # or self.args.flag_enhance).
        return gradio_pipeline_animal.execute_video(*args, **kwargs)


def gpu_wrapped_preview_crop_animal(*args, **kwargs):
    # Expected args: source_image_input, driving_video_input, then crop flags and params, then tab_selection_state
    # The number of args needs to match the inputs list for the button click event.
    # Map args to named variables for clarity
    input_source_image_path = args[0]
    input_driving_video_path = args[1]
    flag_do_crop_input = args[2]
    scale = args[3]
    vx_ratio = args[4]
    vy_ratio = args[5]
    flag_crop_driving_video_input = args[6]
    scale_crop_driving_video = args[7]
    vx_ratio_crop_driving_video = args[8]
    vy_ratio_crop_driving_video = args[9]
    tab_selection = args[10]

    return gradio_pipeline_animal.preview_crop(
        input_source_image_path=input_source_image_path,
        input_driving_video_path=input_driving_video_path,
        flag_do_crop_input=flag_do_crop_input,
        scale=scale,
        vx_ratio=vx_ratio,
        vy_ratio=vy_ratio,
        flag_crop_driving_video_input=flag_crop_driving_video_input,
        scale_crop_driving_video=scale_crop_driving_video,
        vx_ratio_crop_driving_video=vx_ratio_crop_driving_video,
        vy_ratio_crop_driving_video=vy_ratio_crop_driving_video,
        tab_selection=tab_selection
    )


# assets
title_md = "assets/gradio/gradio_title.md"
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"
data_examples_i2v = [
    [osp.join(example_portrait_dir, "s41.jpg"), osp.join(example_video_dir, "d3.mp4"), True, False, False, False],
    [osp.join(example_portrait_dir, "s40.jpg"), osp.join(example_video_dir, "d6.mp4"), True, False, False, False],
    [osp.join(example_portrait_dir, "s25.jpg"), osp.join(example_video_dir, "d19.mp4"), True, False, False, False],
]
data_examples_i2v_pickle = [
    [osp.join(example_portrait_dir, "s25.jpg"), osp.join(example_video_dir, "wink.pkl"), True, False, False, False],
    [osp.join(example_portrait_dir, "s40.jpg"), osp.join(example_video_dir, "talking.pkl"), True, False, False, False],
    [osp.join(example_portrait_dir, "s41.jpg"), osp.join(example_video_dir, "aggrieved.pkl"), True, False, False, False],
]
#################### interface logic ####################

# Define components first
output_image = gr.Image(type="numpy")
output_image_paste_back = gr.Image(type="numpy")
output_video_i2v = gr.Video(autoplay=False)
output_video_concat_i2v = gr.Video(autoplay=False)
output_video_i2v_gif = gr.Image(type="numpy")
output_source_landmarks = gr.Image(type="numpy", label="Source Image with Landmarks")

# New components for animal crop preview
preview_source_cropped_animal = gr.Image(label="Cropped Source Preview (Animal)", type="numpy", visible=True)
preview_driving_cropped_animal = gr.Image(label="Cropped Driving Preview (Animal)", type="numpy", visible=True)


with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")])) as demo:
    gr.HTML(load_description(title_md))

    gr.Markdown(load_description("assets/gradio/gradio_description_upload_animal.md"))
    with gr.Row():
        with gr.Column():
            with gr.Accordion(open=True, label="🐱 Source Animal Image"):
                source_image_input = gr.Image(type="filepath")
                landmark_check_button = gr.Button("🔍 Check Landmarks", variant="secondary", size="sm")
                landmark_check_output = gr.Image(label="Detected Landmarks", type="numpy", show_label=False, interactive=False)
                landmark_check_button.click(
                    fn=gradio_pipeline_animal.visualize_landmarks,
                    inputs=[source_image_input],
                    outputs=[landmark_check_output],
                    show_progress=True
                )
                gr.Examples(
                    examples=[
                        [osp.join(example_portrait_dir, "s25.jpg")],
                        [osp.join(example_portrait_dir, "s30.jpg")],
                        [osp.join(example_portrait_dir, "s31.jpg")],
                        [osp.join(example_portrait_dir, "s32.jpg")],
                        [osp.join(example_portrait_dir, "s33.jpg")],
                        [osp.join(example_portrait_dir, "s39.jpg")],
                        [osp.join(example_portrait_dir, "s40.jpg")],
                        [osp.join(example_portrait_dir, "s41.jpg")],
                        [osp.join(example_portrait_dir, "s38.jpg")],
                        [osp.join(example_portrait_dir, "s36.jpg")],
                    ],
                    inputs=[source_image_input],
                    cache_examples=False,
                )

            with gr.Accordion(open=True, label="Cropping Options for Source Image"):
                with gr.Row():
                    flag_do_crop_input = gr.Checkbox(value=True, label="do crop (source)")
                    scale = gr.Number(value=2.3, label="source crop scale", minimum=1.8, maximum=3.2, step=0.05)
                    vx_ratio = gr.Number(value=0.0, label="source crop x", minimum=-0.5, maximum=0.5, step=0.01)
                    vy_ratio = gr.Number(value=-0.125, label="source crop y", minimum=-0.5, maximum=0.5, step=0.01)

        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("📁 Driving Pickle") as tab_pickle:
                    with gr.Accordion(open=True, label="Driving Pickle"):
                        driving_video_pickle_input = gr.File(type="filepath")
                        gr.Examples(
                            examples=[
                                [osp.join(example_video_dir, "wink.pkl")],
                                [osp.join(example_video_dir, "shy.pkl")],
                                [osp.join(example_video_dir, "aggrieved.pkl")],
                                [osp.join(example_video_dir, "open_lip.pkl")],
                                [osp.join(example_video_dir, "laugh.pkl")],
                                [osp.join(example_video_dir, "talking.pkl")],
                                [osp.join(example_video_dir, "shake_face.pkl")],
                            ],
                            inputs=[driving_video_pickle_input],
                            cache_examples=False,
                        )
                with gr.TabItem("🎞️ Driving Video") as tab_video:
                    with gr.Accordion(open=True, label="Driving Video"):
                        driving_video_input = gr.Video()
                        gr.Examples(
                            examples=[
                                # [osp.join(example_video_dir, "d0.mp4")],
                                # [osp.join(example_video_dir, "d18.mp4")],
                                [osp.join(example_video_dir, "d19.mp4")],
                                [osp.join(example_video_dir, "d14.mp4")],
                                [osp.join(example_video_dir, "d6.mp4")],
                                [osp.join(example_video_dir, "d3.mp4")],
                            ],
                            inputs=[driving_video_input],
                            cache_examples=False,
                        )

                    # tab_selection = gr.Textbox(visible=False) # Old way
                    # tab_pickle.select(lambda: "Pickle", None, tab_selection)
                    # tab_video.select(lambda: "Video", None, tab_selection)

                    # New way using gr.State
                    # Default to 'Pickle' as it's the first tab defined.
                    tab_selection_state = gr.State(value="Pickle")

                    def select_pickle_tab():
                        return "Pickle"
                    def select_video_tab():
                        return "Video"

                    tab_pickle.select(fn=select_pickle_tab, inputs=None, outputs=[tab_selection_state])
                    tab_video.select(fn=select_video_tab, inputs=None, outputs=[tab_selection_state])

            with gr.Accordion(open=True, label="Cropping Options for Driving Video"):
                with gr.Row():
                    flag_crop_driving_video_input = gr.Checkbox(value=False, label="do crop (driving)")
                    scale_crop_driving_video = gr.Number(value=2.2, label="driving crop scale", minimum=1.8, maximum=3.2, step=0.05)
                    vx_ratio_crop_driving_video = gr.Number(value=0.0, label="driving crop x", minimum=-0.5, maximum=0.5, step=0.01)
                    vy_ratio_crop_driving_video = gr.Number(value=-0.1, label="driving crop y", minimum=-0.5, maximum=0.5, step=0.01)

    with gr.Row():
        with gr.Accordion(open=False, label="Animation Options"):
            with gr.Row():
                flag_stitching = gr.Checkbox(value=False, label="stitching (not recommended)")
                flag_remap_input = gr.Checkbox(value=False, label="paste-back (not recommended)")
                driving_multiplier = gr.Number(value=1.0, label="driving multiplier", minimum=0.0, maximum=2.0, step=0.02)
                flag_enhance_input = gr.Checkbox(value=enh_cfg.flag_enhance, label="Enable Upscaling (Enhancement)")

    gr.Markdown(load_description("assets/gradio/gradio_description_animate_clear.md"))
    with gr.Row():
        preview_crop_button_animal = gr.Button("✂️ Preview Crop (Animal)", variant="secondary") # New button for animals
        process_button_animation = gr.Button("🦁 Animate Animal", variant="primary")
    with gr.Row():
        with gr.Column():
            with gr.Accordion(open=True, label="The animated video in the cropped image space"):
                output_video_i2v.render()
        with gr.Column():
            with gr.Accordion(open=True, label="The animated gif in the cropped image space"):
                output_video_i2v_gif.render()
        with gr.Column():
            with gr.Accordion(open=True, label="The animated video"):
                output_video_concat_i2v.render()
        with gr.Column():
            with gr.Accordion(open=True, label="Source Image Landmarks"):
                output_source_landmarks.render()
    with gr.Row():
        process_button_reset = gr.ClearButton([source_image_input, driving_video_input, output_video_i2v, output_video_concat_i2v, output_video_i2v_gif, output_source_landmarks], value="🧹 Clear")

    # Adding row for animal crop previews
    with gr.Row():
        with gr.Column():
            preview_source_cropped_animal.render()
        with gr.Column():
            preview_driving_cropped_animal.render()

    with gr.Accordion("Output GIF (Optional)", open=False):
        with gr.Row():
            with gr.Column():
                pass
            with gr.Column():
                pass

    with gr.Row():
        # Examples
        gr.Markdown("## You could also choose the examples below by one click ⬇️")
    with gr.Row():
        with gr.Tabs():
            with gr.TabItem("📁 Driving Pickle") as tab_video:
                gr.Examples(
                    examples=data_examples_i2v_pickle,
                    fn=gpu_wrapped_execute_video,
                    inputs=[
                        source_image_input,
                        driving_video_pickle_input,
                        flag_do_crop_input,
                        flag_stitching,
                        flag_remap_input,
                        flag_crop_driving_video_input,
                    ],
                    outputs=[output_image, output_image_paste_back, output_video_i2v_gif, output_source_landmarks],
                    examples_per_page=len(data_examples_i2v_pickle),
                    cache_examples=False,
                )
            with gr.TabItem("🎞️ Driving Video") as tab_video:
                gr.Examples(
                    examples=data_examples_i2v,
                    fn=gpu_wrapped_execute_video,
                    inputs=[
                        source_image_input,
                        driving_video_input,
                        flag_do_crop_input,
                        flag_stitching,
                        flag_remap_input,
                        flag_crop_driving_video_input,
                    ],
                    outputs=[output_image, output_image_paste_back, output_video_i2v_gif, output_source_landmarks],
                    examples_per_page=len(data_examples_i2v),
                    cache_examples=False,
                )

    # binding functions for buttons
    process_button_animation.click(
        fn=gpu_wrapped_execute_video,
        inputs=[
            source_image_input,
            driving_video_input,
            driving_video_pickle_input, # driving_video_pickle_input for animals
            flag_do_crop_input,
            flag_remap_input,
            driving_multiplier,
            flag_stitching,
            flag_crop_driving_video_input,
            scale,
            vx_ratio,
            vy_ratio,
            scale_crop_driving_video,
            vx_ratio_crop_driving_video,
            vy_ratio_crop_driving_video,
            tab_selection_state, # Use the state for tab selection
            flag_enhance_input, # Added for enhancement
        ],
        outputs=[
            output_video_i2v,
            output_video_concat_i2v,
            output_video_i2v_gif,
            output_source_landmarks # Added for landmarks
        ],
        show_progress=True
    )

    preview_crop_button_animal.click(
        fn=gpu_wrapped_preview_crop_animal,
        inputs=[
            source_image_input,
            driving_video_input, # Driving video for preview
            flag_do_crop_input,
            scale,
            vx_ratio,
            vy_ratio,
            flag_crop_driving_video_input,
            scale_crop_driving_video,
            vx_ratio_crop_driving_video,
            vy_ratio_crop_driving_video,
            tab_selection_state, # To determine if driving is 'Video'
        ],
        outputs=[
            preview_source_cropped_animal,
            preview_driving_cropped_animal,
        ]
    )

# --- Combine Demos --- (Optional: Combine or launch separately)
# Option 1: Launch separately (simpler)
demo.launch(
    server_port=args.server_port,
    share=args.share,
    server_name=args.server_name
)

# Option 2: Combine using gr.TabbedInterface (More integrated, but requires restructuring)
# If you prefer tabs, the structure of app_animals.py would need significant changes
# to embed both 'demo' and 'demo_landmark_check' Blocks within a TabbedInterface.
# For now, launching separately is the most straightforward way.
