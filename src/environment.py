import numpy as np
import jax.numpy as jnp
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

import jax
from morphology import create_morphology
import mediapy as media
from morphology import default_brittle_star_morphology_specification
import render


# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)
jnp.set_printoptions(precision=3, suppress=True, linewidth=100)


def create_arena(arena_configuration: AquariumArenaConfiguration) -> MJCFAquariumArena:
    arena = MJCFAquariumArena(configuration=arena_configuration)
    return arena


def create_environment(
    morphology_specification: BrittleStarMorphologySpecification,
    arena_configuration: AquariumArenaConfiguration,
    environment_configuration: MuJoCoEnvironmentConfiguration,
    backend: str,
) -> DualMuJoCoEnvironment:
    assert backend in [
        "MJC",
        "MJX",
    ], "Please specify a valid backend; Either 'MJC' or 'MJX'"

    morphology = create_morphology(morphology_specification=morphology_specification)
    arena = create_arena(arena_configuration=arena_configuration)
    if isinstance(
        environment_configuration,
        BrittleStarUndirectedLocomotionEnvironmentConfiguration,
    ):
        env_class = BrittleStarUndirectedLocomotionEnvironment
    elif isinstance(
        environment_configuration, BrittleStarDirectedLocomotionEnvironmentConfiguration
    ):
        env_class = BrittleStarDirectedLocomotionEnvironment
    else:
        env_class = BrittleStarLightEscapeEnvironment

    env = env_class.from_morphology_and_arena(
        morphology=morphology,
        arena=arena,
        configuration=environment_configuration,
        backend=backend,
    )
    return env


def create_example_environment(
    env_type: str = "directed_locomotion",
    simulation_time: int = 5,
    time_scale: int = 2,
    target_distance: float = 3.0,
    num_physics_steps_per_control_step: int = 10,
) -> (
    BrittleStarUndirectedLocomotionEnvironmentConfiguration
    | BrittleStarDirectedLocomotionEnvironmentConfiguration
    | BrittleStarLightEscapeEnvironmentConfiguration
):
    if env_type == "undirected_locomotion":
        return BrittleStarUndirectedLocomotionEnvironmentConfiguration(
            # If this value is > 0 then we will add randomly sampled noise to the initial joint positions and velocities
            joint_randomization_noise_scale=0.0,
            # Visualization mode
            render_mode="rgb_array",
            # Number of seconds per episode
            simulation_time=simulation_time,
            # Number of physics substeps to do per control step
            num_physics_steps_per_control_step=num_physics_steps_per_control_step,
            # Integer factor by which to multiply the original physics timestep of 0.002,
            time_scale=time_scale,
            # Which camera's to render (all the brittle star environments contain 2 cameras: 1 top-down camera and one close-up camera that follows the brittle star),
            camera_ids=[0, 1],
            # Resolution to render with ((height, width) in pixels)
            render_size=(480, 640),
        )
    elif env_type == "directed_locomotion":
        return BrittleStarDirectedLocomotionEnvironmentConfiguration(
            # Distance to put our target at (targets are spawned on a circle around the starting location with this given radius).
            target_distance=target_distance,
            joint_randomization_noise_scale=0.0,
            render_mode="rgb_array",
            simulation_time=simulation_time,
            num_physics_steps_per_control_step=num_physics_steps_per_control_step,
            time_scale=time_scale,
            camera_ids=[0, 1],
            render_size=(480, 640),
            color_contacts=False
        )
    elif env_type == "light_escape":
        return BrittleStarLightEscapeEnvironmentConfiguration(
            # If this value is > 0, we will add perlin noise to the generated light map. Otherwise, the light map is a simple linear gradient.
            #   Please only provide integer factors of 200.
            light_perlin_noise_scale=0,
            joint_randomization_noise_scale=0,
            render_mode="rgb_array",
            simulation_time=simulation_time,
            num_physics_steps_per_control_step=num_physics_steps_per_control_step,
            time_scale=time_scale,
            camera_ids=[0, 1],
            render_size=(480, 640),
        )
    else:
        raise ValueError(f"Invalid environment type: {env_type}")


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
    morphology_specification = default_brittle_star_morphology_specification(
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
        morphology_specification=morphology_specification,
        arena_configuration=arena_configuration,
        environment_configuration=environment_configuration,
        backend=backend,
    )

    # Initialize random number generator
    if backend == "MJC":
        rng = np.random.RandomState(seed)
        state = env.reset(rng=rng)
    else:
        rng = jax.random.PRNGKey(seed=seed)
        state = jax.jit(env.reset)(rng=rng)


    # Reset environment
    

    return env, state, environment_configuration


def run_simulation(env, state, environment_configuration):
    """Run the simulation and collect frames"""
    mjc_frames = []

    # Get initial frame
    frame = env.render(state=state)
    processed_frame = render.post_render(
        render_output=frame, environment_configuration=environment_configuration
    )
    mjc_frames.append(processed_frame)

    while not (state.terminated | state.truncated):
        # TODO: measure observations to determine action
        # Sample random actions to pass to the environment
        action = env.action_space.sample()
        # print('Action:', action)
        state = env.step(state=state, action=action)  # Update state
        # print('State:', state)
        # TODO: collect observations and rewards
        processed_frame = render.post_render(
            env.render(state=state), environment_configuration
        )
        mjc_frames.append(processed_frame)

    return mjc_frames



def test():
    NUM_MJX_ENVIRONMENTS = 2  # set this to the number of CUDA cores that you have available

  

    mjx_vectorized_env, state, environment_configuration = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=5,
        num_segments_per_arm=[2, 0, 0, 2, 0],
        backend="MJX",
    )


    

    mjx_action_rng, mjx_vectorized_env_rng = jax.random.split(jax.random.PRNGKey(0), 2)
    mjx_vectorized_env_rng = jnp.array(jax.random.split(mjx_vectorized_env_rng, NUM_MJX_ENVIRONMENTS))

    mjx_vectorized_step = jax.jit(jax.vmap(mjx_vectorized_env.step))
    mjx_vectorized_reset = jax.jit(jax.vmap(mjx_vectorized_env.reset))
    mjx_vectorized_action_sample = jax.jit(jax.vmap(mjx_vectorized_env.action_space.sample))

    mjx_vectorized_state = mjx_vectorized_reset(rng=mjx_vectorized_env_rng)

    mjx_frames = []
    while not jnp.any(mjx_vectorized_state.terminated | mjx_vectorized_state.truncated):
        mjx_action_rng, *sub_rngs = jnp.array(jax.random.split(mjx_action_rng, NUM_MJX_ENVIRONMENTS + 1))
        action = mjx_vectorized_action_sample(rng=jnp.array(sub_rngs))

        mjx_vectorized_state = mjx_vectorized_step(state=mjx_vectorized_state, action=action)
        mjx_frames.append(
                render.post_render(
                        mjx_vectorized_env.render(state=mjx_vectorized_state), mjx_vectorized_env.environment_configuration
                        )
                )
        print(mjx_vectorized_state.observations["xy_distance_to_target"][0])
        
    print(f"Simulation complete with {len(mjx_frames)} frames!")
    output_dir = "./training_progress"


    #render.save_frame_samples(mjx_frames, output_dir=output_dir, sample_rate=10)
    # video_path = f"{output_dir}/brittle_star_simulation.mp4"
    # render.create_animation(mjx_frames, output_path=video_path)

if __name__ == "__main__":

    #test()
    # Initialize simulation
    env, state, environment_configuration = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=5,
        num_segments_per_arm=4,
        backend="MJC",
    )
    env._env._mj_model.cam_pos[0] *= 0.35  

    # Get initial frame and display
    initial_frame = render.post_render(
        env.render(state=state), environment_configuration
    )

    render.visualize_initial_frame(initial_frame)

    # Run full simulation
    print("Running simulation...")
    mjc_frames = run_simulation(env, state, environment_configuration)
    print(f"Simulation complete with {len(mjc_frames)} frames!")

    # Create output directory and save selected frames
    output_dir = "simulation_output"
    #render.save_frame_samples(mjc_frames, output_dir=output_dir, sample_rate=10)
    render.visualize_initial_frame(mjc_frames[0])

    # # Create and save animation
    # video_path = f"{output_dir}/brittle_star_simulation.mp4"
    # render.create_animation(mjc_frames, output_path=video_path)
    # print(f"Saved animation to {video_path}")

    # # Play the video
    # render.play_video(video_path)
