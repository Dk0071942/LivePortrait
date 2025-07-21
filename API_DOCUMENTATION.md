# LivePortrait API Documentation

## Table of Contents

1. [Core Classes](#core-classes)
2. [Pipeline APIs](#pipeline-apis)
3. [Neural Network APIs](#neural-network-apis)
4. [Utility APIs](#utility-apis)
5. [Configuration APIs](#configuration-apis)
6. [Usage Examples](#usage-examples)

## Core Classes

### LivePortraitPipeline

Main orchestrator for portrait animation pipeline.

```python
class LivePortraitPipeline(object):
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig)
```

#### Methods

##### execute(args: ArgumentConfig) -> None
Execute the complete animation pipeline.

**Parameters:**
- `args`: Configuration object containing all execution parameters

**Workflow:**
1. Load source and driving inputs
2. Perform face cropping and preprocessing  
3. Extract features and create animations
4. Generate output video/image

##### make_motion_template(I_lst, c_eyes_lst, c_lip_lst, **kwargs) -> dict
Create a reusable motion template from driving video.

**Parameters:**
- `I_lst`: List of driving video frames
- `c_eyes_lst`: Eye retargeting coefficients
- `c_lip_lst`: Lip retargeting coefficients
- `**kwargs`: Additional parameters (output_fps, etc.)

**Returns:**
- Dictionary containing motion data and metadata

##### prepare_videos(driving_rgb_lst, driving_rgb_lst_256, **kwargs) -> dict
Preprocess driving videos for animation.

**Parameters:**
- `driving_rgb_lst`: Original resolution frames
- `driving_rgb_lst_256`: Cropped 256x256 frames
- `**kwargs`: Processing options

**Returns:**
- Preprocessed video data dictionary

### LivePortraitWrapper

Neural network interface for model inference.

```python
class LivePortraitWrapper(object):
    def __init__(self, inference_cfg: InferenceConfig)
```

#### Methods

##### forward(x_s, x_d, R_s, R_d, expression_s, expression_d, **kwargs)
Main inference method for animation generation.

**Parameters:**
- `x_s`: Source keypoints (B, num_kp, 3)
- `x_d`: Driving keypoints (B, num_kp, 3)
- `R_s`: Source rotation matrix
- `R_d`: Driving rotation matrix
- `expression_s`: Source expression parameters
- `expression_d`: Driving expression parameters

**Returns:**
- Generated output frame

##### extract_feature_3d(x_s)
Extract 3D appearance features from source.

**Parameters:**
- `x_s`: Source image tensor

**Returns:**
- 3D feature representation

##### transform_keypoint(x_s, x_d, R_s=None, R_d=None, **kwargs)
Transform keypoints for motion transfer.

**Parameters:**
- `x_s`: Source keypoints
- `x_d`: Driving keypoints  
- `R_s`: Optional source rotation
- `R_d`: Optional driving rotation

**Returns:**
- Transformed keypoints

##### stitching(x_s, x_d)
Apply stitching network for boundary smoothing.

**Parameters:**
- `x_s`: Source features
- `x_d`: Driving features

**Returns:**
- Stitching coefficients

### Cropper

Face detection and cropping utilities.

```python
class Cropper(object):
    def __init__(self, crop_cfg: CropConfig)
```

#### Methods

##### crop_single_image(img_rgb, **kwargs)
Process a single image for face cropping.

**Parameters:**
- `img_rgb`: Input RGB image
- `**kwargs`: Additional options

**Returns:**
- Dictionary containing:
  - `img_crop_256`: Cropped face image
  - `lmk_crop_256`: Facial landmarks
  - `M_c2o`: Crop-to-original transformation matrix

##### crop_driving_video(driving_rgb_lst, **kwargs)
Process driving video for consistent cropping.

**Parameters:**
- `driving_rgb_lst`: List of RGB frames
- `**kwargs`: Processing options

**Returns:**
- List of cropped frames and metadata

##### get_retargeting_matrix(img_rgb, **kwargs)
Compute retargeting transformation matrix.

**Parameters:**
- `img_rgb`: Input image
- `**kwargs`: Retargeting options

**Returns:**
- Retargeting matrix and coefficients

## Pipeline APIs

### GradioPipeline

Extended pipeline for Gradio web interface.

```python
class GradioPipeline(LivePortraitPipeline):
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig)
```

#### Additional Methods

##### execute_video_retargeting(input_video, retarget_eyes, retarget_lips)
Process video with eye/lip retargeting controls.

**Parameters:**
- `input_video`: Source video path
- `retarget_eyes`: Eye control parameters
- `retarget_lips`: Lip control parameters

**Returns:**
- Retargeted video output

### LivePortraitPipelineAnimal

Specialized pipeline for animal portraits.

```python
class LivePortraitPipelineAnimal(LivePortraitPipeline):
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig)
```

#### Differences from Human Pipeline
- Uses X-Pose for animal keypoint detection
- Modified cropping strategy for animal faces
- Different motion retargeting approach

## Neural Network APIs

### AppearanceFeatureExtractor

ResNet-based feature extraction network.

```python
class AppearanceFeatureExtractor(nn.Module):
    def __init__(self, **kwargs)
```

#### Methods

##### forward(x)
Extract appearance features.

**Parameters:**
- `x`: Input image tensor (B, 3, H, W)

**Returns:**
- Feature tensor (B, C, D, H', W')

### MotionExtractor

Keypoint and expression extraction network.

```python
class MotionExtractor(nn.Module):
    def __init__(self, **kwargs)
```

#### Methods

##### forward(x)
Extract motion parameters.

**Parameters:**
- `x`: Input image tensor

**Returns:**
- Dictionary containing:
  - `kp`: Keypoints (B, num_kp, 3)
  - `expression`: Expression parameters
  - `rotation`: Rotation matrix

### WarpingNetwork

Dense motion field generation.

```python
class WarpingNetwork(nn.Module):
    def __init__(self, **kwargs)
```

#### Methods

##### forward(feature, x_s, x_d)
Generate dense motion fields.

**Parameters:**
- `feature`: Source features
- `x_s`: Source keypoints
- `x_d`: Driving keypoints

**Returns:**
- Dense motion field and occlusion map

### SPADEGenerator

High-quality image synthesis network.

```python
class SPADEGenerator(nn.Module):
    def __init__(self, **kwargs)
```

#### Methods

##### forward(feature, warped_feature)
Generate output image.

**Parameters:**
- `feature`: Original features
- `warped_feature`: Warped features

**Returns:**
- Generated image (B, 3, H*2, W*2)

### StitchingRetargetingNetwork

Boundary smoothing and control network.

```python
class StitchingRetargetingNetwork(nn.Module):
    def __init__(self, **kwargs)
```

#### Components
- `stitching`: MLP for boundary smoothing
- `eye`: MLP for eye retargeting
- `lip`: MLP for lip retargeting

## Utility APIs

### Video Processing

```python
# src/utils/video.py

def images2video(images, output_path, fps=25, crf=15)
"""Convert image sequence to video"""

def concat_frames(frame_list, axis=1)
"""Concatenate frames for comparison"""

def add_audio_to_video(video_path, audio_path, output_path)
"""Add audio track to video"""

def get_fps(video_path)
"""Get video frame rate"""
```

### Image Processing

```python
# src/utils/io.py

def load_image_rgb(image_path)
"""Load image in RGB format"""

def resize_to_limit(img, max_dim=1280, division=2)
"""Resize image with constraints"""

def dump(obj, path)
"""Save object as pickle file"""

def load(path)
"""Load pickle file"""
```

### Face Detection

```python
# src/utils/face_analysis_diy.py

class FaceAnalysisDIY:
    def __init__(self, **kwargs)
    
    def detect_faces(self, img)
    """Detect faces in image"""
    
    def get_face_landmarks(self, img, face)
    """Extract facial landmarks"""
```

### Camera Utilities

```python
# src/utils/camera.py

def get_rotation_matrix(pitch, yaw, roll)
"""Compute 3D rotation matrix"""

def project_keypoints(kp_3d, camera_matrix)
"""Project 3D keypoints to 2D"""
```

## Configuration APIs

### InferenceConfig

```python
@dataclass
class InferenceConfig(PrintableConfig):
    # Model paths
    models_config: str
    checkpoint_F: str
    checkpoint_M: str
    checkpoint_G: str
    checkpoint_W: str
    checkpoint_S: str
    
    # Inference settings
    flag_use_half_precision: bool = True
    device_id: int = 0
    
    # Animation options
    driving_option: Literal["pose-friendly", "expression-friendly"]
    driving_multiplier: float = 1.0
    animation_region: Literal["exp", "pose", "lip", "eyes", "all"]
    
    # Processing flags
    flag_stitching: bool = True
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_relative_motion: bool = True
```

### CropConfig

```python
@dataclass
class CropConfig(PrintableConfig):
    # Detection settings
    det_thresh: float = 0.05
    max_faces: int = 5
    
    # Cropping parameters
    dsize: int = 256
    scale: float = 2.3
    vx_ratio: float = 0.0
    vy_ratio: float = -0.125
    
    # Processing options
    flag_force_cpu: bool = False
```

### ArgumentConfig

```python
@dataclass
class ArgumentConfig(PrintableConfig):
    # Input/Output paths
    source: str
    driving: str
    output_dir: str = "./animations"
    
    # Processing options
    flag_crop_driving_video: bool = False
    scale_crop_driving_video: float = 2.2
    
    # Advanced options
    flag_pasteback: bool = True
    flag_do_crop: bool = True
    flag_do_rot: bool = True
```

## Usage Examples

### Basic Animation

```python
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.live_portrait_pipeline import LivePortraitPipeline

# Initialize configurations
inference_cfg = InferenceConfig()
crop_cfg = CropConfig()
args = ArgumentConfig(
    source="path/to/source.jpg",
    driving="path/to/driving.mp4"
)

# Create pipeline
pipeline = LivePortraitPipeline(inference_cfg, crop_cfg)

# Execute animation
pipeline.execute(args)
```

### Motion Template Creation

```python
# Load driving video
driving_frames = load_video("driving.mp4")

# Create motion template
template = pipeline.make_motion_template(
    I_lst=driving_frames,
    c_eyes_lst=eye_coefficients,
    c_lip_lst=lip_coefficients,
    output_fps=25
)

# Save template
dump(template, "motion_template.pkl")
```

### Custom Feature Extraction

```python
# Initialize wrapper
wrapper = LivePortraitWrapper(inference_cfg)

# Extract features
source_img = load_image_rgb("source.jpg")
features = wrapper.extract_feature_3d(source_img)

# Extract keypoints
motion_data = wrapper.motion_extractor(source_img)
keypoints = motion_data['kp']
expression = motion_data['expression']
```

### Eye and Lip Retargeting

```python
# Enable retargeting
inference_cfg.flag_eye_retargeting = True
inference_cfg.flag_lip_retargeting = True

# Set retargeting parameters
eye_ratio = 1.2  # Amplify eye movements
lip_ratio = 0.8  # Reduce lip movements

# Apply during inference
output = wrapper.forward(
    x_s, x_d, R_s, R_d,
    expression_s, expression_d,
    eye_ratio=eye_ratio,
    lip_ratio=lip_ratio
)
```

### Batch Processing

```python
# Process multiple source images
source_images = ["img1.jpg", "img2.jpg", "img3.jpg"]
driving_template = load("template.pkl")

for source in source_images:
    args = ArgumentConfig(
        source=source,
        driving=driving_template,
        output_dir="./batch_output"
    )
    pipeline.execute(args)
```

---

*This API documentation covers the main interfaces of LivePortrait. For implementation details, refer to the source code and inline documentation.*