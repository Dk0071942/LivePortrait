# LivePortrait Project Documentation

> **Efficient Portrait Animation with Stitching and Retargeting Control**

This document provides comprehensive technical documentation for the LivePortrait project, an advanced portrait animation system that transfers motion from driving videos to source portraits while maintaining high-quality output and precise control.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Neural Network Modules](#neural-network-modules)
5. [Processing Pipeline](#processing-pipeline)
6. [API Reference](#api-reference)
7. [Configuration System](#configuration-system)
8. [Project Structure](#project-structure)
9. [Development Guide](#development-guide)
10. [Performance Specifications](#performance-specifications)

## Project Overview

### Description
LivePortrait is a state-of-the-art portrait animation framework that enables realistic motion transfer from driving videos to source portraits. It supports both human and animal portraits with high-quality output and real-time performance.

### Key Features
- **Efficient Animation**: Real-time portrait animation with minimal computational overhead
- **Stitching Control**: Seamless blending of animated regions with original portraits
- **Retargeting Control**: Precise control over eye and lip movements
- **Multi-Modal Support**: Works with images, videos, and motion templates
- **Privacy Protection**: Motion template caching for privacy-preserving animation
- **Cross-Platform**: Support for Linux, Windows, and macOS (with limitations)
- **Dual Mode**: Separate pipelines for humans and animals

### Technical Innovation
- Novel stitching mechanism for natural boundary transitions
- Advanced retargeting modules for fine-grained control
- Efficient motion template system for fast inference
- SPADE-based generator for high-quality synthesis

## Architecture

### System Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  User Interface Layer                │
├─────────────────┬──────────────┬────────────────────┤
│   Gradio Web UI │  CLI Interface│   API Endpoints   │
│    (app.py)     │(inference.py) │  (future work)    │
└────────┬────────┴──────────────┴────────────────────┘
         │
┌────────▼────────────────────────────────────────────┐
│              Pipeline Orchestration Layer            │
├─────────────────┬───────────────┬───────────────────┤
│LivePortraitPipe │ GradioPipeline│ AnimalPipeline    │
└────────┬────────┴───────────────┴───────────────────┘
         │
┌────────▼────────────────────────────────────────────┐
│            Core Processing Components                │
├──────────────┬────────────────┬─────────────────────┤
│LivePortrait  │    Cropper     │  Motion Template    │
│   Wrapper    │   Component    │     System          │
└──────┬───────┴────────┬───────┴─────────────────────┘
       │                │
┌──────▼────────────────▼─────────────────────────────┐
│          Neural Network Modules (F,M,W,G,S&R)       │
└─────────────────────────────────────────────────────┘
```

### Component Relationships

1. **User Interface Layer**: Entry points for user interaction
2. **Pipeline Layer**: Orchestrates the entire animation workflow
3. **Core Components**: Handles processing logic and data flow
4. **Neural Network Layer**: Deep learning models for animation

## Core Components

### LivePortraitPipeline
**Location**: `src/live_portrait_pipeline.py`

The main orchestrator for human portrait animation.

**Key Methods**:
- `execute()`: Main entry point for animation
- `make_motion_template()`: Creates reusable motion templates
- `calc_retargeting_ratio()`: Computes retargeting parameters
- `prepare_videos()`: Preprocesses input videos

**Responsibilities**:
- Input validation and preprocessing
- Workflow orchestration
- Output generation and post-processing

### LivePortraitWrapper
**Location**: `src/live_portrait_wrapper.py`

Neural network interface and inference engine.

**Key Methods**:
- `forward()`: Main inference method
- `extract_feature_3d()`: Extracts 3D facial features
- `transform_keypoint()`: Applies motion transformations
- `stitching()`: Performs boundary smoothing

**Components**:
- Appearance Feature Extractor (F)
- Motion Extractor (M)  
- Warping Module (W)
- SPADE Generator (G)
- Stitching/Retargeting Module (S&R)

### Cropper
**Location**: `src/utils/cropper.py`

Face detection and preprocessing component.

**Key Features**:
- InsightFace-based detection
- 68-point landmark extraction
- Automatic face alignment
- Multi-face support

**Methods**:
- `crop_single_image()`: Process single images
- `crop_driving_video()`: Process driving videos
- `calc_crop_limit()`: Compute cropping boundaries

### GradioPipeline
**Location**: `src/gradio_pipeline.py`

Web interface pipeline extending LivePortraitPipeline.

**Additional Features**:
- Real-time preview
- Interactive controls
- Progress tracking
- Error handling

## Neural Network Modules

### Module F: Appearance Feature Extractor
**Purpose**: Extracts deep appearance features from source portraits

**Architecture**:
- ResNet-based backbone
- Multi-scale feature extraction
- 3D feature representation
- Channel reshaping for efficiency

**Specifications**:
```yaml
input_channels: 3
block_expansion: 64
num_down_blocks: 2
max_features: 512
reshape_channel: 32
reshape_depth: 16
num_resblocks: 6
```

### Module M: Motion Extractor
**Purpose**: Detects facial keypoints and expression parameters

**Architecture**:
- ConvNeXtV2 backbone
- 21 keypoint detection
- Expression parameter extraction
- Rotation matrix computation

**Specifications**:
```yaml
num_keypoints: 21
backbone: convnextv2_tiny
output: [keypoints, expressions, rotation]
```

### Module W: Warping Module
**Purpose**: Creates dense motion fields for animation

**Architecture**:
- Dense motion network
- Occlusion estimation
- Multi-scale processing
- Motion field generation

**Specifications**:
```yaml
num_keypoints: 21
block_expansion: 64
max_features: 512
estimate_occlusion: true
```

### Module G: SPADE Generator
**Purpose**: Synthesizes high-quality output frames

**Architecture**:
- SPADE normalization
- Progressive upsampling
- Multi-scale synthesis
- 2x upscaling (256→512)

**Specifications**:
```yaml
upscale_factor: 2
block_expansion: 64
max_features: 512
num_down_blocks: 2
```

### Module S&R: Stitching & Retargeting
**Purpose**: Smooth boundaries and enable fine control

**Components**:
- **Stitching Network**: Face boundary smoothing
- **Eye Retargeting**: Precise eye movement control
- **Lip Retargeting**: Accurate lip sync control

**Specifications**:
```yaml
stitching:
  input_size: 126
  hidden_sizes: [128, 128, 64]
  output_size: 65
lip:
  input_size: 65
  hidden_sizes: [128, 128, 64]
  output_size: 63
eye:
  input_size: 66
  hidden_sizes: [256, 256, 128, 128, 64]
  output_size: 63
```

## Processing Pipeline

### Phase 1: Input Processing
```python
1. Load inputs (source & driving)
2. Validate formats and compatibility
3. Detect faces and landmarks
4. Crop to standard resolution (256x256)
```

### Phase 2: Feature Extraction
```python
1. Extract source appearance features (F)
2. Extract driving motion features (M)
3. Compute keypoints and expressions
4. Generate rotation matrices
```

### Phase 3: Motion Processing
```python
1. Create/load motion templates
2. Apply motion multipliers
3. Smooth motion trajectories
4. Handle retargeting parameters
```

### Phase 4: Animation Loop
```python
for each frame:
    1. Transform source keypoints
    2. Blend expressions
    3. Generate dense motion fields (W)
    4. Synthesize output frame (G)
    5. Apply stitching if enabled (S)
    6. Collect results
```

### Phase 5: Post-Processing
```python
1. Paste back to original resolution
2. Apply temporal smoothing
3. Add audio if present
4. Generate output files
```

## API Reference

### LivePortraitPipeline

```python
class LivePortraitPipeline:
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        """Initialize pipeline with configuration"""
        
    def execute(self, args: ArgumentConfig) -> None:
        """Execute full animation pipeline"""
        
    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs) -> dict:
        """Create reusable motion template"""
        
    def prepare_videos(self, driving_rgb_lst, driving_rgb_lst_256, **kwargs) -> dict:
        """Preprocess video inputs"""
```

### LivePortraitWrapper

```python
class LivePortraitWrapper:
    def forward(self, x_s, x_d, R_s, R_d, expression_s, expression_d, **kwargs):
        """Main inference method"""
        
    def extract_feature_3d(self, x):
        """Extract 3D features from input"""
        
    def transform_keypoint(self, x_s, x_d, R_s=None, R_d=None, **kwargs):
        """Apply keypoint transformations"""
        
    def stitching(self, x_s, x_d):
        """Perform boundary stitching"""
```

### Cropper

```python
class Cropper:
    def crop_single_image(self, img_rgb, **kwargs):
        """Process single image"""
        
    def crop_driving_video(self, driving_rgb_lst, **kwargs):
        """Process driving video"""
        
    def get_retargeting_matrix(self, img_rgb, **kwargs):
        """Compute retargeting parameters"""
```

## Configuration System

### InferenceConfig
Main configuration for model inference.

**Key Parameters**:
- `models_config`: Path to model architecture config
- `checkpoint_*`: Paths to pretrained weights
- `flag_use_half_precision`: Enable FP16 inference
- `device_id`: GPU device selection
- `driving_option`: "pose-friendly" or "expression-friendly"
- `animation_region`: Control animation regions

### CropConfig
Configuration for face detection and cropping.

**Key Parameters**:
- `dsize`: Target resolution (256)
- `scale`: Face cropping scale
- `max_faces`: Maximum faces to detect
- `detect_thresh`: Detection confidence threshold

### ArgumentConfig
User-facing configuration options.

**Key Parameters**:
- `source`: Input source path
- `driving`: Driving video/template path
- `output_dir`: Output directory
- `flag_*`: Various processing flags

## Project Structure

```
LivePortrait/
├── src/
│   ├── config/              # Configuration modules
│   │   ├── argument_config.py
│   │   ├── base_config.py
│   │   ├── crop_config.py
│   │   ├── inference_config.py
│   │   └── models.yaml
│   ├── modules/             # Neural network modules
│   │   ├── appearance_feature_extractor.py
│   │   ├── convnextv2.py
│   │   ├── dense_motion.py
│   │   ├── motion_extractor.py
│   │   ├── spade_generator.py
│   │   ├── stitching_retargeting_network.py
│   │   └── warping_network.py
│   ├── utils/               # Utility functions
│   │   ├── dependencies/    # External dependencies
│   │   │   ├── insightface/ # Face detection
│   │   │   └── XPose/       # Animal keypoints
│   │   ├── cropper.py       # Face cropping
│   │   ├── camera.py        # Camera utilities
│   │   ├── video.py         # Video processing
│   │   └── helper.py        # Helper functions
│   ├── live_portrait_pipeline.py      # Main pipeline
│   ├── live_portrait_wrapper.py       # Model wrapper
│   └── gradio_pipeline.py             # Web interface
├── assets/                  # Resources and examples
│   ├── docs/               # Documentation
│   ├── examples/           # Example inputs
│   └── gradio/            # UI resources
├── pretrained_weights/     # Model checkpoints
├── app.py                  # Gradio interface
├── inference.py            # CLI interface
└── requirements.txt        # Dependencies
```

## Development Guide

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/KwaiVGI/LivePortrait
cd LivePortrait

# Create conda environment
conda create -n LivePortrait python=3.10
conda activate LivePortrait

# Install dependencies
pip install -r requirements.txt
```

### Adding New Features

1. **New Motion Control**:
   - Extend `stitching_retargeting_module.py`
   - Add control parameters to `InferenceConfig`
   - Update pipeline logic in `live_portrait_pipeline.py`

2. **Custom Preprocessing**:
   - Extend `Cropper` class
   - Add configuration to `CropConfig`
   - Update pipeline preprocessing

3. **New Output Format**:
   - Modify `video.py` utilities
   - Update `execute()` in pipeline
   - Add format options to config

### Code Style Guidelines

- Follow PEP 8 conventions
- Use type hints for function signatures
- Document all public methods
- Keep functions focused and modular

## Performance Specifications

### Speed Benchmarks
**Hardware**: NVIDIA RTX 4090

| Component | Time (ms) | FPS |
|-----------|-----------|-----|
| Face Detection | 15-20 | 50-66 |
| Feature Extraction | 8-12 | 83-125 |
| Motion Transfer | 20-30 | 33-50 |
| Frame Generation | 25-35 | 28-40 |
| **Total Pipeline** | 68-97 | 10-15 |

### Resource Usage
- **GPU Memory**: 2-6GB (depending on batch size)
- **System RAM**: 8GB recommended
- **Storage**: 10GB for full installation

### Optimization Tips
1. Enable half precision: `flag_use_half_precision=True`
2. Use motion templates for repeated animations
3. Enable torch.compile for 20-30% speedup
4. Batch process multiple frames when possible

---

*This documentation provides a comprehensive technical overview of the LivePortrait system. For the latest updates and community contributions, visit the [GitHub repository](https://github.com/KwaiVGI/LivePortrait).*