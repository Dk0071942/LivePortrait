import os
import glob
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension # Combined imports
from setuptools import find_packages, setup

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # Check for FORCE_CUDA environment variable
    force_cuda_build = os.environ.get('FORCE_CUDA', '0') == '1' # [1]

    # Decide whether to build with CUDA
    # Condition: (FORCE_CUDA=1 and CUDA_HOME is found) OR (torch.cuda.is_available() and CUDA_HOME is found)
    if (force_cuda_build and CUDA_HOME is not None) or \
       (torch.cuda.is_available() and CUDA_HOME is not None): # [1]

        if force_cuda_build and not torch.cuda.is_available():
            print("INFO: FORCE_CUDA=1 detected. CUDA build will be attempted despite torch.cuda.is_available() being False.")
        else:
            print("INFO: CUDA build enabled (torch.cuda.is_available() is True and CUDA_HOME is set).")

        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("WARNING: CUDA not available or FORCE_CUDA not set/CUDA_HOME not found. Building CPU-only version. GPU functionality may not work at runtime.") # [1]
        pass

    # Ensure paths in 'sources' are absolute or relative to extensions_dir as intended by original script
    # The original script did: sources = [os.path.join(extensions_dir, s) for s in sources]
    # This seems to imply the glob results were not full paths. Let's clarify this.
    # Assuming glob gives paths relative to extensions_dir or full paths that need to be processed.
    # For safety, let's ensure they are joined correctly if they are not already absolute.
    processed_sources = []
    for s_file in sources:
        if not os.path.isabs(s_file):
            processed_sources.append(os.path.join(extensions_dir, s_file))
        else:
            processed_sources.append(s_file)

    # The original code for sources was:
    # sources = [os.path.join(extensions_dir, s) for s in sources]
    # This implies that 'main_file', 'source_cpu', 'source_cuda' might have been just filenames.
    # Let's re-evaluate the source collection based on the original structure for safety:

    _sources = glob.glob(os.path.join(extensions_dir, "*.cpp")) + \
               glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

    if extension == CUDAExtension: # If CUDA is enabled by the logic above
        _sources += glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))


    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            _sources, # Use the re-evaluated _sources list
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension}, # Corrected to use BuildExtension directly
)
