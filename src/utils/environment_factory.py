"""Factory module for creating simulation environments."""
import numpy as np
import jax.numpy as jnp
from typing import Union, Tuple
from biorobot.brittle_star.environment.light_escape.shared import (
    BrittleStarLightEscapeEnvironmentConfiguration,
)
from biorobot.brittle_star.environment.directed_locomotion.shared import (
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.environment.undirected_locomotion.shared import (
    BrittleStarUndirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.mjcf.arena.aquarium import (
    AquariumArenaConfiguration,
    MJCFAquariumArena,
)
from biorobot.brittle_star.mjcf.morphology.specification.specification import (
    BrittleStarMorphologySpecification,
)
from moojoco.environment.dual import DualMuJoCoEnvironment
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from biorobot.brittle_star.environment.undirected_locomotion.dual import (
    BrittleStarUndirectedLocomotionEnvironment,
)
from biorobot.brittle_star.environment.directed_locomotion.dual import (
    BrittleStarDirectedLocomotionEnvironment,
)
from biorobot.brittle_star.environment.light_escape.dual import (
    BrittleStarLightEscapeEnvironment,
)

from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology


# Configure numpy and jax.numpy for more readable output
np.set_printoptions(precision=3, suppress=True, linewidth=100)
jnp.set_printoptions(precision=3, suppress=True, linewidth=100)


def create_arena(arena_config: AquariumArenaConfiguration) -> MJCFAquariumArena:
    """Create an aquarium arena for the simulation.
    
    Args:
        arena_config: Configuration for the aquarium arena
        
    Returns:
        Configured aquarium arena
    """
    return MJCFAquariumArena(configuration=arena_config)


def create_morphology(morphology_spec: BrittleStarMorphologySpecification) -> MJCFBrittleStarMorphology:
    """Create a brittle star morphology from specification.
    
    Args:
        morphology_spec: Specification for the brittle star morphology
        
    Returns:
        Brittle star morphology
    """
    return MJCFBrittleStarMorphology(specification=morphology_spec)



def create_environment(
    morphology_spec: BrittleStarMorphologySpecification,
    arena_config: AquariumArenaConfiguration,
    env_config: MuJoCoEnvironmentConfiguration,
    backend: str,
) -> DualMuJoCoEnvironment:
    """Create a simulation environment with specified morphology, arena and configuration.
    
    Args:
        morphology_spec: Specification for the brittle star morphology
        arena_config: Configuration for the simulation arena
        env_config: Configuration for the environment
        backend: Either 'MJC' or 'MJX'
        
    Returns:
        Configured simulation environment
        
    Raises:
        ValueError: If an invalid backend is provided
    """
    if backend not in ["MJC", "MJX"]:
        raise ValueError("Backend must be either 'MJC' or 'MJX'")

    morphology = create_morphology(morphology_spec=morphology_spec)
    arena = create_arena(arena_config=arena_config)
    
    # Select the appropriate environment class based on configuration
    if isinstance(env_config, BrittleStarUndirectedLocomotionEnvironmentConfiguration):
        env_class = BrittleStarUndirectedLocomotionEnvironment
    elif isinstance(env_config, BrittleStarDirectedLocomotionEnvironmentConfiguration):
        env_class = BrittleStarDirectedLocomotionEnvironment
    else:
        env_class = BrittleStarLightEscapeEnvironment

    # Create the environment
    env = env_class.from_morphology_and_arena(
        morphology=morphology,
        arena=arena,
        configuration=env_config,
        backend=backend,
    )
    return env


def create_default_environment_config(
    env_type: str = "directed_locomotion",
    simulation_time: int = 5,
    time_scale: int = 2,
    target_distance: float = 3.0,
    num_physics_steps_per_control_step: int = 10,
) -> Union[
    BrittleStarUndirectedLocomotionEnvironmentConfiguration,
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
    BrittleStarLightEscapeEnvironmentConfiguration,
]:
    """Create a default environment configuration.
    
    Args:
        env_type: Type of environment ('undirected_locomotion', 'directed_locomotion', or 'light_escape')
        simulation_time: Duration of simulation in seconds
        time_scale: Time scale for the simulation
        target_distance: Distance to the target
        num_physics_steps_per_control_step: Physics steps per control step
        
    Returns:
        Environment configuration
        
    Raises:
        ValueError: If an invalid environment type is provided
    """
    base_config = {
        "joint_randomization_noise_scale": 0.0,
        "render_mode": "rgb_array",
        "simulation_time": simulation_time,
        "num_physics_steps_per_control_step": num_physics_steps_per_control_step,
        "time_scale": time_scale,
    }
    
    if env_type == "undirected_locomotion":
        return BrittleStarUndirectedLocomotionEnvironmentConfiguration(**base_config)
    elif env_type == "directed_locomotion":
        return BrittleStarDirectedLocomotionEnvironmentConfiguration(
            **base_config,
            target_distance=target_distance,
            camera_ids=[0, 1],
            render_size=(480, 640),
        )
    elif env_type == "light_escape":
        return BrittleStarLightEscapeEnvironmentConfiguration(
            **base_config,
            num_light_sources=1,
            light_strength=2.0,
            camera_ids=[0, 1],
            render_size=(480, 640),
        )
    else:
        raise ValueError(
            "Environment type must be one of 'undirected_locomotion', 'directed_locomotion', or 'light_escape'"
        )


def create_default_arena_config(
    arena_size: Tuple[float, float] = (1.5, 1.5),
    attach_target: bool = True,
    wall_height: float = 1.5,
) -> AquariumArenaConfiguration:
    """Create a default arena configuration.
    
    Args:
        arena_size: Size of the arena (width, depth)
        attach_target: Whether to attach a target to the arena
        wall_height: Height of the arena walls
        
    Returns:
        Arena configuration
    """
    return AquariumArenaConfiguration(
        size=arena_size,
        sand_ground_color=False,
        attach_target=attach_target,
        wall_height=wall_height,
        wall_thickness=0.1,
    )
