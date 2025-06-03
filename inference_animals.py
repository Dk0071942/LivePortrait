# coding: utf-8

"""
The entrance of animal
"""

import os
import os.path as osp
import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.config.enhancement_config import EnhancementConfig
from src.live_portrait_pipeline_animal import LivePortraitPipelineAnimal
import traceback


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")


def main():
    print("[DEBUG inference_animals.py main] Script started.")
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    print(f"[DEBUG inference_animals.py main] Arguments parsed: {args}")

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    print("[DEBUG inference_animals.py main] Checking FFmpeg...")
    if not fast_check_ffmpeg():
        print("[ERROR inference_animals.py main] FFmpeg check failed.")
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )
    print("[DEBUG inference_animals.py main] FFmpeg check passed.")

    print("[DEBUG inference_animals.py main] Checking arguments...")
    fast_check_args(args)
    print("[DEBUG inference_animals.py main] Argument check passed.")

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)
    enh_cfg = partial_fields(EnhancementConfig, args.__dict__)
    print("[DEBUG inference_animals.py main] Configurations created.")

    print("[DEBUG inference_animals.py main] Initializing LivePortraitPipelineAnimal...")
    live_portrait_pipeline_animal = LivePortraitPipelineAnimal(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg,
        enh_cfg=enh_cfg
    )
    print("[DEBUG inference_animals.py main] LivePortraitPipelineAnimal initialized.")

    print("[DEBUG inference_animals.py main] Attempting to call execute method...")
    try:
        # run
        live_portrait_pipeline_animal.execute(args)
        print("[DEBUG inference_animals.py main] Execute method finished successfully.")
    except Exception as e:
        print(f"[ERROR inference_animals.py main] Exception during execute: {e}")
        print(f"[ERROR inference_animals.py main] Traceback: {traceback.format_exc()}")
        raise
    print("[DEBUG inference_animals.py main] Script finished.")


if __name__ == "__main__":
    main()
