import os
import torch
import numpy as np
import torchvision
from PIL import Image
import cv2
import subprocess
from tqdm import tqdm
import time

# Import RealESRGANer and RRDBNet architecture for RealESRGAN_x4plus.
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Import the new config class
from ..config.enhancement_config import EnhancementConfig

class ImageUpscale:
    def __init__(self, enh_cfg: EnhancementConfig, model_name: str = "RealESRGAN_x4plus", half: bool = False):
        """
        Initializes the RealESRGAN based upscaling model.

        Args:
            enh_cfg (EnhancementConfig): Configuration object with enhancement settings.
            model_name (str): The model name; corresponds to the weight file (without extension).
            half (bool): Whether to use FP16 mode for performance.
        """
        self.enh_cfg = enh_cfg # Store the enhancement config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netscale =  4 # x4 upscaling factor for RealESRGAN
        self.model_name = model_name
        model_filename = f"{model_name}.pth"

        # Define the model architecture matching the defaults of RealESRGAN_x4plus
        self.model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                  num_block=23, num_grow_ch=32, scale=self.netscale)

        # Construct model path relative to this script.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "upscale_models")
        self.model_path = os.path.join(model_dir, model_filename)
        print(f"Using model path: {self.model_path}")

        try:
            self.upsampler = RealESRGANer(
                scale=self.netscale,
                model_path=self.model_path,
                model=self.model_arch,
                tile=self.enh_cfg.upscaler_tile, # Use tile from config
                tile_pad=self.enh_cfg.upscaler_tile_pad, # Use tile_pad from config
                pre_pad=0,
                half=half,
                device=self.device
            )
            print("RealESRGANer initialized successfully.")
        except Exception as e:
            print("Error initializing RealESRGANer:", e)
            print("Please ensure the model file exists and is compatible.")
            self.upsampler = None

    def upscale(self, input_image_path: str, outscale: int = None) -> Image.Image:
        """
        Upscales an input image (PIL-based) using the RealESRGANer pipeline.

        Args:
            input_image_path (str): Path to the input image.
            outscale (int, optional): The scaling factor for the output image.

        Returns:
            PIL.Image: The upscaled image.
        """
        if self.upsampler is None:
            print("Upsampler not initialized.")
            return None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, input_image_path)
        print(f"Loading image from: {image_path}")
        try:
            img_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print("Error loading image:", e)
            return None

        # Convert from PIL (RGB) to NumPy array (BGR for RealESRGAN)
        img_np = np.array(img_pil)
        img_np_bgr = img_np[:, :, ::-1]

        if outscale is None:
            outscale = self.netscale
        try:
            output_np_bgr, _ = self.upsampler.enhance(img_np_bgr, outscale=outscale)
            output_np_rgb = output_np_bgr[:, :, ::-1]
            sr_image = Image.fromarray(output_np_rgb)
            print("Image enhancement successful.")
            return sr_image
        except Exception as e:
            print("An error occurred during enhancement:", e)
            return None

    def enhance_image(self, decoded_latents: torch.Tensor, sharpness_factor: float = 1.5) -> torch.Tensor:
        """
        Enhances the image quality using a sharpening filter.

        Expects an input tensor of shape (N, C, H, W) in the range [-1, 1] and returns a tensor of
        the same shape after enhancement.

        Args:
            decoded_latents (torch.Tensor): Input tensor with values in [-1, 1].
            sharpness_factor (float): Factor for adjusting image sharpness.

        Returns:
            torch.Tensor: Enhanced tensor with values in [-1, 1].
        """
        # Convert input from [-1, 1] to [0, 1]
        enhanced_latents = (decoded_latents / 2 + 0.5).clamp(0, 1)
        # Apply the sharpening filter using torchvision's adjust_sharpness.
        enhanced_latents = torchvision.transforms.functional.adjust_sharpness(enhanced_latents, sharpness_factor)
        # Convert back to [-1, 1]
        enhanced_latents = enhanced_latents * 2 - 1
        return enhanced_latents

    def enhance_and_upscale(self, input_tensor: torch.Tensor, sharpness_factor: float = 1.5) -> torch.Tensor:
        """
        Accepts an input image tensor, applies a sharpening filter and upscales it using RealESRGAN.
        The upscaled image is then resized back to the original spatial dimensions so that the output
        tensor has the same shape as the input.

        Assumptions:
          - The input tensor is of shape (N, C, H, W) with values in [-1, 1].
          - This function processes each image in the batch individually.

        Args:
            input_tensor (torch.Tensor): Input image tensor with shape (N, C, H, W) and range [-1, 1].
            sharpness_factor (float): The factor for the sharpening filter.

        Returns:
            torch.Tensor: The enhanced and upscaled tensor with the same shape as the input, on the same device.
        """
        if self.upsampler is None:
            print("Upsampler not initialized.")
            return None

        # Ensure the input tensor is on the proper device.
        input_tensor = input_tensor.to(self.device)
        # First, apply the sharpening enhancement.
        enhanced_tensor = self.enhance_image(input_tensor, sharpness_factor=sharpness_factor)
        N, C, H, W = enhanced_tensor.shape
        output_tensors = []

        for i in range(N):
            # Extract one image: shape (C, H, W), range [-1, 1]
            img_tensor = enhanced_tensor[i]
            # Convert to [0, 1] range.
            img_tensor_0_1 = (img_tensor + 1) / 2
            # Convert tensor (C, H, W) to PIL image.
            pil_image = torchvision.transforms.functional.to_pil_image(img_tensor_0_1.cpu())
            # Convert PIL image to numpy array (RGB) and then to BGR (RealESRGAN expects BGR).
            np_image = np.array(pil_image)
            np_image_bgr = np_image[:, :, ::-1]

            try:
                # Upscale using RealESRGANer. The output will be upscaled by self.netscale.
                upscaled_np_bgr, _ = self.upsampler.enhance(np_image_bgr, outscale=self.netscale)
            except Exception as e:
                print("Error during RealESRGAN upscaling:", e)
                return None

            # Convert the upscaled image from BGR to RGB.
            upscaled_np_rgb = upscaled_np_bgr[:, :, ::-1]
            upscaled_pil = Image.fromarray(upscaled_np_rgb)
            # Resize the upscaled image back to the original dimensions.
            downscaled_pil = upscaled_pil.resize((W, H), Image.BICUBIC)
            # Convert downscaled image to tensor; resulting tensor is in range [0, 1].
            downscaled_tensor = torchvision.transforms.functional.to_tensor(downscaled_pil)
            # Convert to [-1, 1] range.
            final_tensor = downscaled_tensor * 2 - 1
            # Add back a batch dimension.
            output_tensors.append(final_tensor.unsqueeze(0))

        # Concatenate along the batch dimension and move back to the device.
        return torch.cat(output_tensors, dim=0).to(self.device)

    def process_video(self, input_video_path: str) -> str | None:
        """
        Upscales a video file frame by frame using RealESRGAN and preserves audio.

        Args:
            input_video_path (str): Path to the input video file.

        Returns:
            str | None: Path to the upscaled video file, or None if an error occurs.
        """
        if self.upsampler is None:
            print("Upsampler not initialized.")
            return None

        if not os.path.exists(input_video_path):
            print(f"Input video not found: {input_video_path}")
            return None

        # Create output path
        output_dir = os.path.dirname(input_video_path)
        base_name = os.path.basename(input_video_path)
        name, ext = os.path.splitext(base_name)
        output_video_path = os.path.join(output_dir, f"{name}_upscaled_x{self.enh_cfg.enhance_outscale}{ext}")
        temp_video_path = os.path.join(output_dir, f"temp_{name}_upscaled{ext}") # Without audio initially

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {input_video_path}")
            return None

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Target dimensions
        target_width = width * self.enh_cfg.enhance_outscale
        target_height = height * self.enh_cfg.enhance_outscale

        # Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4
        out_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_width, target_height))

        print(f"Processing video: {input_video_path} -> {output_video_path}")
        start_time = time.time()

        for _ in tqdm(range(frame_count), desc="Upscaling video frames"):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            try:
                # Upscale frame (RealESRGAN expects BGR)
                output_frame_bgr, _ = self.upsampler.enhance(frame_bgr, outscale=self.enh_cfg.enhance_outscale)
                out_writer.write(output_frame_bgr)
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Continue processing other frames if possible

        cap.release()
        out_writer.release()

        end_time = time.time()
        print(f"Video frames upscaled in {end_time - start_time:.2f} seconds.")

        # --- Audio Handling using FFmpeg ---
        # Check if source video has audio
        has_audio = False
        try:
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0',
                input_video_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            if 'audio' in probe_result.stdout.lower():
                has_audio = True
        except Exception as e:
            print(f"Could not probe audio for {input_video_path}: {e}. Assuming no audio.")
            has_audio = False

        if has_audio:
            print("Source video has audio. Merging audio into upscaled video using ffmpeg...")
            try:
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output file if it exists
                    '-i', temp_video_path,      # Input video (upscaled, no audio)
                    '-i', input_video_path,     # Input audio source
                    '-map', '0:v:0',            # Map video from first input
                    '-map', '1:a:0',            # Map audio from second input
                    '-c:v', 'copy',             # Copy video stream (no re-encoding)
                    '-c:a', 'aac',              # Encode audio to aac (common)
                    '-shortest',                # Finish encoding when the shortest stream ends
                    output_video_path
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"Audio successfully merged into: {output_video_path}")
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                return output_video_path
            except subprocess.CalledProcessError as e:
                print(f"Error during ffmpeg audio merge: {e}")
                print(f"FFmpeg stdout: {e.stdout.decode()}")
                print(f"FFmpeg stderr: {e.stderr.decode()}")
                print(f"Upscaled video (without audio) saved at: {temp_video_path}")
                # Rename temp file if merge fails but temp exists
                if os.path.exists(temp_video_path):
                    os.rename(temp_video_path, output_video_path)
                    print(f"Renamed temp file to {output_video_path}")
                    return output_video_path
                return None # Indicate failure
            except Exception as e:
                print(f"An unexpected error occurred during audio merge: {e}")
                # Attempt to preserve the video-only file
                if os.path.exists(temp_video_path):
                    os.rename(temp_video_path, output_video_path)
                    print(f"Renamed temp file to {output_video_path} due to unexpected error.")
                    return output_video_path
                return None
        else:
            print("Source video has no audio stream. Renaming temp file.")
            # Just rename the temporary file to the final output path
            if os.path.exists(temp_video_path):
                os.rename(temp_video_path, output_video_path)
                return output_video_path
            else:
                print("Error: Temporary upscaled video file not found.")
                return None
