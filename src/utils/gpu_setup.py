"""GPU and MuJoCo initialization utilities."""
import logging
import os
import subprocess


def configure_nvidia_egl():
    """Configure NVIDIA EGL for MuJoCo rendering on GPU."""
    nvidia_icd_config_path = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    
    if not os.path.exists(nvidia_icd_config_path):
        with open(nvidia_icd_config_path, 'w') as config_file:
            config_file.write(
                """{
                    "file_format_version" : "1.0.0",
                    "ICD" : {
                        "library_path" : "libEGL_nvidia.so.0"
                    }
                }"""
            )


def setup_gpu_rendering():
    """Configure environment for GPU rendering with MuJoCo."""
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ["DISPLAY"] = ""
    
    # Check GPU availability
    if subprocess.run('nvidia-smi', capture_output=True, check=False).returncode != 0:
        raise RuntimeError('Cannot communicate with GPU.')
    
    configure_nvidia_egl()
    
    # Verify GPU recognition by JAX
    import jax
    gpu_devices = jax.devices('gpu')
    print(f"Available GPU devices: {gpu_devices}")
    return gpu_devices


def verify_mujoco_installation():
    """Verify that MuJoCo is correctly installed and functional."""
    try:
        import mujoco
        mujoco.MjModel.from_xml_string('<mujoco/>')
        print('MuJoCo installation successful.')
        return mujoco
    except Exception as error:
        raise RuntimeError(
            'MuJoCo installation failed. Check the output above for more information.\n'
            'If using a hosted runtime, make sure you enable GPU acceleration.'
        ) from error


def initialize_system():
    """Initialize the system for simulation and return the mujoco module."""
    try:
        setup_gpu_rendering()
    except Exception as error:
        logging.warning(f"GPU initialization failed. Using CPU. Error: {error}")
    
    return verify_mujoco_installation()
