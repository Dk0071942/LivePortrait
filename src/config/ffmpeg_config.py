# coding: utf-8
"""
FFmpeg configuration for consistent video encoding across the LivePortrait project.

This module provides centralized FFmpeg encoding parameters to ensure consistent
video quality and compatibility across all video generation operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class FFmpegConfig:
    """Centralized FFmpeg configuration for consistent video encoding.
    
    This configuration ensures all video outputs use the same high-quality
    encoding parameters for consistency and compatibility.
    """
    
    # Video codec configuration
    video_codec: str = "libx264"
    preset: str = "slow"  # Quality/speed tradeoff: ultrafast to veryslow
    crf: int = 18  # Constant Rate Factor (0-51, lower = better quality)
    
    # Pixel format and color space
    pix_fmt: str = "yuv420p"  # Most compatible pixel format
    color_space: str = "bt709"  # HD color space
    color_primaries: str = "bt709"  # HD color primaries
    color_trc: str = "bt709"  # HD transfer characteristics
    
    # Additional encoding flags
    movflags: str = "+faststart"  # Enable fast start for web playback
    
    # Video parameters
    default_fps: int = 30
    macro_block_size: int = 2  # For dimension compatibility
    
    # Audio configuration
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    
    # Hardware acceleration (disabled by default for compatibility)
    use_hardware_accel: bool = False
    hardware_accel_type: str = "cuda"  # cuda, qsv, vaapi, videotoolbox
    
    # Profile settings for compatibility
    profile: str = "high"  # baseline, main, high
    level: str = "4.1"  # Compatibility level
    
    # Additional custom parameters
    additional_params: List[str] = field(default_factory=list)
    
    def get_output_params(self, include_audio: bool = False) -> Dict[str, str]:
        """Get imageio-ffmpeg output parameters dictionary.
        
        Args:
            include_audio: Whether to include audio codec parameters
            
        Returns:
            Dictionary of FFmpeg output parameters for imageio
        """
        params = {
            'codec': self.video_codec,
            'preset': self.preset,
            'crf': str(self.crf),
            'pix_fmt': self.pix_fmt,
            'movflags': self.movflags,
        }
        
        # Add color space parameters
        params.update({
            'color_primaries': self.color_primaries,
            'color_trc': self.color_trc,
            'colorspace': self.color_space,
        })
        
        # Add profile and level
        params['profile:v'] = self.profile
        params['level'] = self.level
        
        # Add audio codec if needed
        if include_audio:
            params['acodec'] = self.audio_codec
            params['audio_bitrate'] = self.audio_bitrate
            
        return params
    
    def get_ffmpeg_args(self, include_color_space: bool = True) -> List[str]:
        """Get FFmpeg command line arguments.
        
        Args:
            include_color_space: Whether to include color space parameters
            
        Returns:
            List of FFmpeg command line arguments
        """
        args = [
            '-c:v', self.video_codec,
            '-preset', self.preset,
            '-crf', str(self.crf),
            '-pix_fmt', self.pix_fmt,
            '-movflags', self.movflags,
            '-profile:v', self.profile,
            '-level', self.level,
        ]
        
        if include_color_space:
            args.extend([
                '-color_primaries', self.color_primaries,
                '-color_trc', self.color_trc,
                '-colorspace', self.color_space,
            ])
            
        # Add any additional custom parameters
        if self.additional_params:
            args.extend(self.additional_params)
            
        return args
    
    def get_gif_palette_args(self) -> List[str]:
        """Get FFmpeg arguments for GIF palette generation.
        
        Returns:
            List of FFmpeg arguments for palette generation
        """
        return ['-vf', 'fps=10,scale=320:-1:flags=lanczos,palettegen']
    
    def get_gif_encoding_args(self) -> List[str]:
        """Get FFmpeg arguments for GIF encoding.
        
        Returns:
            List of FFmpeg arguments for GIF encoding
        """
        return ['-filter_complex', 'fps=10,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse']


# Default configuration instance
default_ffmpeg_config = FFmpegConfig()