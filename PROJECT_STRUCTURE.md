# LivePortrait Project Structure

## Visual Directory Tree

```
LivePortrait/
│
├── 📄 app.py                           # Main Gradio web interface
├── 📄 app_animals.py                   # Gradio interface for animal mode
├── 📄 inference.py                     # Command-line interface
├── 📄 speed.py                         # Performance benchmarking tool
│
├── 📄 requirements.txt                 # Python dependencies
├── 📄 requirements_base.txt            # Core dependencies
├── 📄 requirements_macOS.txt           # macOS-specific dependencies
│
├── 📄 Dockerfile                       # Docker container configuration
├── 📄 LICENSE                          # Apache 2.0 license
├── 📄 readme.md                        # Main project documentation
├── 📄 readme_zh_cn.md                  # Chinese documentation
│
├── 📁 src/                             # Source code directory
│   ├── 📁 config/                      # Configuration modules
│   │   ├── 📄 __init__.py
│   │   ├── 📄 argument_config.py       # CLI argument configuration
│   │   ├── 📄 base_config.py           # Base configuration class
│   │   ├── 📄 crop_config.py           # Face cropping configuration
│   │   ├── 📄 enhancement_config.py    # Enhancement settings
│   │   ├── 📄 inference_config.py      # Model inference configuration
│   │   └── 📄 models.yaml              # Neural network architecture specs
│   │
│   ├── 📁 modules/                     # Neural network modules
│   │   ├── 📄 __init__.py
│   │   ├── 📄 appearance_feature_extractor.py  # Module F: Feature extraction
│   │   ├── 📄 convnextv2.py                   # ConvNeXtV2 backbone
│   │   ├── 📄 dense_motion.py                 # Dense motion field generation
│   │   ├── 📄 motion_extractor.py             # Module M: Motion extraction
│   │   ├── 📄 spade_generator.py              # Module G: SPADE synthesis
│   │   ├── 📄 stitching_retargeting_network.py # Module S&R: Stitching/control
│   │   ├── 📄 util.py                         # Module utilities
│   │   └── 📄 warping_network.py              # Module W: Warping
│   │
│   ├── 📁 utils/                       # Utility functions
│   │   ├── 📄 __init__.py
│   │   ├── 📄 animal_landmark_runner.py # Animal keypoint detection
│   │   ├── 📄 camera.py                # Camera transformation utilities
│   │   ├── 📄 check_windows_port.py    # Windows port availability check
│   │   ├── 📄 crop.py                  # Image cropping utilities
│   │   ├── 📄 cropper.py               # Main face cropping class
│   │   ├── 📄 face_analysis_diy.py     # Face detection wrapper
│   │   ├── 📄 filter.py                # Temporal filtering
│   │   ├── 📄 helper.py                # General helper functions
│   │   ├── 📄 human_landmark_runner.py # Human landmark detection
│   │   ├── 📄 image_upscale.py         # Image enhancement
│   │   ├── 📄 io.py                    # Input/output utilities
│   │   ├── 📄 retargeting_utils.py     # Retargeting calculations
│   │   ├── 📄 rprint.py                # Rich print utilities
│   │   ├── 📄 timer.py                 # Performance timing
│   │   ├── 📄 video.py                 # Video processing utilities
│   │   ├── 📄 viz.py                   # Visualization tools
│   │   │
│   │   ├── 📁 dependencies/            # External dependencies
│   │   │   ├── 📁 insightface/         # Face detection library
│   │   │   │   ├── 📁 app/             # Face analysis applications
│   │   │   │   ├── 📁 data/            # Data utilities and assets
│   │   │   │   ├── 📁 model_zoo/       # Pre-trained models
│   │   │   │   └── 📁 utils/           # InsightFace utilities
│   │   │   │
│   │   │   └── 📁 XPose/               # Animal pose detection
│   │   │       ├── 📁 config_model/    # Model configurations
│   │   │       ├── 📁 models/          # XPose neural networks
│   │   │       └── 📁 util/            # XPose utilities
│   │   │
│   │   ├── 📁 resources/               # Static resources
│   │   │   ├── 📄 clip_embedding_68.pkl # 68-point landmark embeddings
│   │   │   ├── 📄 clip_embedding_9.pkl  # 9-point landmark embeddings
│   │   │   ├── 📄 lip_array.pkl        # Lip movement data
│   │   │   └── 🖼️ mask_template.png    # Face mask template
│   │   │
│   │   └── 📁 upscale_models/          # Enhancement models
│   │       └── 📄 RealESRGAN_x4plus.pth # 4x upscaling model
│   │
│   ├── 📄 gradio_pipeline.py           # Gradio web interface pipeline
│   ├── 📄 live_portrait_pipeline.py    # Main human animation pipeline
│   ├── 📄 live_portrait_pipeline_animal.py # Animal animation pipeline
│   └── 📄 live_portrait_wrapper.py     # Neural network wrapper
│
├── 📁 assets/                          # Project assets
│   ├── 📁 docs/                        # Documentation
│   │   ├── 📁 changelog/               # Version changelogs
│   │   │   ├── 📄 2024-07-10.md
│   │   │   ├── 📄 2024-07-19.md
│   │   │   ├── 📄 2024-07-24.md
│   │   │   ├── 📄 2024-08-02.md
│   │   │   ├── 📄 2024-08-05.md
│   │   │   ├── 📄 2024-08-06.md
│   │   │   ├── 📄 2024-08-19.md
│   │   │   └── 📄 2025-01-01.md
│   │   │
│   │   ├── 📄 animal-mouth-quality-analysis.md
│   │   ├── 📄 directory-structure.md
│   │   ├── 📄 gpu-dependencies-fix.md
│   │   ├── 📄 how-to-install-ffmpeg.md
│   │   ├── 📄 speed.md                 # Performance benchmarks
│   │   ├── 📄 system-workflow.md       # System architecture
│   │   ├── 🖼️ system-workflow_mermaid.svg
│   │   └── 📄 video-based-mouth-enhancement.md
│   │
│   ├── 📁 examples/                    # Example inputs
│   │   ├── 📁 source/                  # Source portraits
│   │   └── 📁 driving/                 # Driving videos/templates
│   │
│   ├── 📁 gradio/                      # Gradio UI resources
│   │   ├── 📄 gradio_description_*.md  # UI text content
│   │   └── 📄 gradio_title.md
│   │
│   └── 📁 goPet/                       # Animal-specific assets
│
├── 📁 pretrained_weights/              # Model checkpoints (not in repo)
│   ├── 📁 insightface/                 # Face detection models
│   │   └── 📁 models/
│   │       └── 📁 buffalo_l/
│   │           ├── 📄 2d106det.onnx    # 106-point detection
│   │           └── 📄 det_10g.onnx     # Face detection
│   │
│   ├── 📁 liveportrait/                # Human animation models
│   │   ├── 📁 base_models/
│   │   │   ├── 📄 appearance_feature_extractor.pth
│   │   │   ├── 📄 motion_extractor.pth
│   │   │   ├── 📄 spade_generator.pth
│   │   │   └── 📄 warping_module.pth
│   │   │
│   │   ├── 📄 landmark.onnx            # Landmark detection
│   │   └── 📁 retargeting_models/
│   │       └── 📄 stitching_retargeting_module.pth
│   │
│   └── 📁 liveportrait_animals/        # Animal animation models
│       ├── 📁 base_models/
│       ├── 📁 retargeting_models/
│       └── 📄 xpose.pth                # Animal pose detection
│
└── 📁 animations/                      # Output directory (generated)
```

## Module Organization

### Core Components

```
┌─────────────────────────────────────────────────┐
│                  Entry Points                    │
├──────────────┬──────────────┬───────────────────┤
│   app.py     │ inference.py │  app_animals.py   │
│  (Web UI)    │   (CLI)      │  (Animal UI)      │
└──────┬───────┴──────┬───────┴────────┬──────────┘
       │              │                 │
       ▼              ▼                 ▼
┌──────────────────────────────────────────────────┐
│              Pipeline Layer                       │
├─────────────────┬────────────────────────────────┤
│ GradioPipeline  │  LivePortraitPipeline(Animal)  │
└────────┬────────┴────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│          Core Processing Components               │
├──────────────┬──────────────┬────────────────────┤
│ LivePortrait │   Cropper    │  Motion Template   │
│   Wrapper    │              │     System         │
└──────┬───────┴──────┬───────┴────────────────────┘
       │              │
       ▼              ▼
┌──────────────────────────────────────────────────┐
│        Neural Network Modules (F,M,W,G,S&R)      │
└──────────────────────────────────────────────────┘
```

### Neural Network Architecture

```
Input Image
    │
    ▼
┌─────────────────┐     ┌─────────────────┐
│   Module F      │     │   Module M      │
│ Feature Extract │     │ Motion Extract  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │   Keypoints &   │
         │              │   Expressions   │
         │              └────────┬────────┘
         │                       │
         ▼                       ▼
┌──────────────────────────────────────────┐
│            Module W                       │
│         Warping Network                   │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│            Module G                       │
│         SPADE Generator                   │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│          Module S&R                       │
│    Stitching & Retargeting               │
└────────────────┬─────────────────────────┘
                 │
                 ▼
            Output Frame
```

## Key File Purposes

### Entry Points
- **app.py**: Gradio web interface for interactive use
- **inference.py**: Command-line tool for batch processing
- **app_animals.py**: Specialized interface for animal portraits

### Configuration Files
- **inference_config.py**: Model paths, inference settings
- **crop_config.py**: Face detection and cropping parameters
- **argument_config.py**: User-facing CLI arguments
- **models.yaml**: Neural network architecture specifications

### Pipeline Components
- **live_portrait_pipeline.py**: Main orchestration logic
- **live_portrait_wrapper.py**: Neural network interface
- **gradio_pipeline.py**: Web UI extensions
- **cropper.py**: Face detection and preprocessing

### Neural Network Modules
- **appearance_feature_extractor.py**: ResNet-based feature extraction
- **motion_extractor.py**: Keypoint and expression detection
- **warping_network.py**: Dense motion field generation
- **spade_generator.py**: High-quality image synthesis
- **stitching_retargeting_network.py**: Boundary smoothing and control

### Utilities
- **video.py**: Video I/O and processing
- **camera.py**: 3D transformations
- **face_analysis_diy.py**: InsightFace wrapper
- **retargeting_utils.py**: Eye/lip control calculations

## Data Flow

```
1. User Input
   ├── Source Portrait (Image/Video)
   └── Driving Input (Video/Template)
        │
2. Preprocessing
   ├── Face Detection (InsightFace)
   ├── Landmark Extraction
   └── Cropping & Alignment
        │
3. Feature Extraction
   ├── Appearance Features (Module F)
   └── Motion Features (Module M)
        │
4. Animation Generation
   ├── Motion Transfer (Module W)
   ├── Frame Synthesis (Module G)
   └── Boundary Smoothing (Module S&R)
        │
5. Post-processing
   ├── Paste Back
   ├── Temporal Smoothing
   └── Audio Addition
        │
6. Output
   ├── Animated Video
   └── Motion Template
```

---

*This structure visualization provides a comprehensive overview of the LivePortrait project organization, making it easy to navigate and understand the codebase.*