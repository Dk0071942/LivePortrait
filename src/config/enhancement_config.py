# coding: utf-8

"""
Config dataclass used for image enhancement (upscaling).
"""

from dataclasses import dataclass
from .base_config import PrintableConfig

@dataclass(repr=False)  # use repr from PrintableConfig
class EnhancementConfig(PrintableConfig):
    """Configuration settings for image enhancement."""
    flag_enhance: bool = True       # Flag to enable/disable final video enhancement (upscaling)
    enhance_outscale: int = 4      # Upscaling factor (e.g., 2, 4). Effective only if flag_enhance is True.
    # RealESRGAN specific parameters
    upscaler_tile: int = 0           # Tile size for RealESRGAN upsampler (0 disables tiling)
    upscaler_tile_pad: int = 10    # Tile padding for RealESRGAN upsampler
