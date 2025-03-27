import logging
import os
import subprocess
import jax
from functools import partial

from biorobot.brittle_star.environment.directed_locomotion.shared import (
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
)
import numpy as np
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from biorobot.brittle_star.environment.directed_locomotion.dual import (
    BrittleStarDirectedLocomotionEnvironment,
)
from typing import List
import mediapy as media
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import (
    default_brittle_star_morphology_specification,
)
from biorobot.brittle_star.mjcf.arena.aquarium import (
    AquariumArenaConfiguration,
    MJCFAquariumArena,
)


import environment as env

DEBUG = True

#################### Initialization code ####################
try:
    if subprocess.run("nvidia-smi").returncode:
        raise RuntimeError("Cannot communicate with GPU.")

    # Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
    # This is usually installed as part of an Nvidia driver package, but the Colab
    # kernel doesn't install its driver via APT, and as a result the ICD is missing.
    # (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
    NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
        with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
            f.write(
                """{
                            "file_format_version" : "1.0.0",
                            "ICD" : {
                                "library_path" : "libEGL_nvidia.so.0"
                            }
                        }
                        """
            )

    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    print("Setting environment variable to use GPU rendering:")

    # xla_flags = os.environ.get('XLA_FLAGS', '')
    # xla_flags += ' --xla_gpu_triton_gemm_any=True'
    # os.environ['XLA_FLAGS'] = xla_flags

    # Check if jax finds the GPU
    import jax

    print(jax.devices("gpu"))
except Exception:
    logging.warning("Failed to initialize GPU. Everything will run on the cpu.")

try:
    print("Checking that the mujoco installation succeeded:")
    import mujoco

    mujoco.MjModel.from_xml_string("<mujoco/>")
except Exception as e:
    raise e from RuntimeError(
        "Something went wrong during installation. Check the shell output above "
        "for more information.\n"
        "If using a hosted Colab runtime, make sure you enable GPU acceleration "
        'by going to the Runtime menu and selecting "Choose runtime type".'
    )

print("MuJoCo installation successful.")


#################### Creating the experimental enviriment ####################
morphology_specification = default_brittle_star_morphology_specification(
    num_arms=5, num_segments_per_arm=3, use_p_control=True, use_torque_control=False
)
arena_configuration = AquariumArenaConfiguration(
    size=(1.5, 1.5),
    sand_ground_color=False,
    attach_target=True,
    wall_height=1.5,
    wall_thickness=0.1,
)
environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(
    target_distance=1.2,
    joint_randomization_noise_scale=0.0,
    render_mode="rgb_array",
    simulation_time=20,
    num_physics_steps_per_control_step=10,
    time_scale=2,
    camera_ids=[0, 1],
    render_size=(480, 640),
)

experimental_env = env.create_environment(
    morphology_specification, arena_configuration, environment_configuration, "MJX"
)

rng = jax.random.PRNGKey(seed=0)

# Fix the target location
env_fixed_target_reset_fn = jax.jit(
    partial(experimental_env.reset, target_position=(-1.25, 0.75, 0.0))
)
env_step_fn = jax.jit(experimental_env.step)


#################### Some Printing functions ####################
def post_render(
    render_output: List[np.ndarray],
    environment_configuration: MuJoCoEnvironmentConfiguration,
) -> np.ndarray:
    if render_output is None:
        # Temporary workaround until https://github.com/google-deepmind/mujoco/issues/1379 is fixed
        return None

    num_cameras = len(environment_configuration.camera_ids)
    num_envs = len(render_output) // num_cameras

    if num_cameras > 1:
        # Horizontally stack frames of the same environment
        frames_per_env = np.array_split(render_output, num_envs)
        render_output = [
            np.concatenate(env_frames, axis=1) for env_frames in frames_per_env
        ]

    # Vertically stack frames of different environments
    render_output = np.concatenate(render_output, axis=0)

    return np.array(render_output)[:, :, ::-1]  # RGB to BGR


def show_video(
    images: List[np.ndarray | None], sim_time: float, path: str | None = None
) -> str | None:
    if path:
        media.write_video(path=path, images=[img for img in images if img is not None])
    return media.show_video(
        images=[img for img in images if img is not None], fps=len(images) // sim_time
    )


if DEBUG:
    rng, sub_rng = jax.random.split(rng, 2)
    env_state = env_fixed_target_reset_fn(sub_rng)
    print("Observation space:")
    print(experimental_env.observation_space)
    print()
    print("Action space:")
    print(experimental_env.action_space)
    print()
    print("Info:")
    print(env_state.info)
    print()
    media.show_image(
        post_render(
            experimental_env.render(env_state),
            environment_configuration=experimental_env.environment_configuration,
        )
    )
