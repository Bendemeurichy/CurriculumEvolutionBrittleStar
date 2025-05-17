"""Environment creation and management for brittle star simulations."""
import numpy as np
import jax
from biorobot.brittle_star.mjcf.arena.aquarium import (
    AquariumArenaConfiguration,
)
from utils.environment_factory import (
    create_arena,
    create_environment,
    create_default_environment_config as create_example_environment,
)
from morphology import create_morphology, create_brittle_star_morphology_specification

def initialize_simulation(
    env_type="directed_locomotion",
    num_arms=5,
    num_segments_per_arm=4,
    backend="MJC",
    simulation_time=5,
    time_scale=2,
    target_distance=3.0,
    num_physics_steps_per_control_step=10,
    seed=0,
):
    """Initialize the brittle star simulation environment"""
    morphology_specification = create_brittle_star_morphology_specification(
        num_arms=num_arms,
        num_segments_per_arm=num_segments_per_arm,
        use_p_control=True,
        use_torque_control=False,
    )

    arena_configuration = AquariumArenaConfiguration(
        size=(10, 5),
        sand_ground_color=False,
        attach_target=True,
        wall_height=1.5,
        wall_thickness=0.1,
    )
    environment_configuration = create_example_environment(
        env_type=env_type,
        simulation_time=simulation_time,
        time_scale=time_scale,
        target_distance=target_distance,
        num_physics_steps_per_control_step=num_physics_steps_per_control_step,
    )

    env = create_environment(
        morphology_spec=morphology_specification,
        arena_config=arena_configuration,
        env_config=environment_configuration,
        backend=backend,
    )

    if backend == "MJC":
        rng = np.random.RandomState(seed)
        state = env.reset(rng=rng)
    else:
        rng = jax.random.PRNGKey(seed=seed)
        state = jax.jit(env.reset)(rng=rng)

    return env, state, environment_configuration

__all__ = [
    "create_arena",
    "create_environment",
    "create_example_environment",
    "create_morphology",
    "create_brittle_star_morphology_specification"
]
