import jax.numpy as jnp
from moojoco.environment.mjx_env import MJXEnvState

# ! axis orientation is X,Y = horizontal, depth, Z = vertical


def observe(env_state: MJXEnvState) -> dict:
    return {
        "disk_position": get_disk_position(env_state),
        "disk_direction": get_disk_direction(env_state),
        "disk_velocity": get_disk_velocity(env_state),
        "disk_angular_velocity": get_disk_angular_velocity(env_state),
        "distance_to_target": get_distance_to_target(env_state),
        "direction_to_target": get_direction_to_target(env_state),
        "joint_positions": [get_joint_positions(env_state, arm) for arm in range(5)],
        "joint_velocities": [get_joint_velocities(env_state, arm) for arm in range(5)],
        "joint_torques": [get_joint_torques(env_state, arm) for arm in range(5)],
        "actuator_force": [get_actuator_force(env_state, arm) for arm in range(5)],
    }


def get_disk_position(env_state: MJXEnvState) -> jnp.ndarray:
    return env_state.observations["disk_position"][:2]


def get_disk_direction(env_state: MJXEnvState) -> jnp.ndarray:
    return env_state.observations["disk_rotation"][2]


def get_disk_velocity(env_state: MJXEnvState) -> jnp.ndarray:
    return env_state.observations["disk_linear_velocity"][:2]


def get_disk_angular_velocity(env_state: MJXEnvState) -> jnp.ndarray:
    return env_state.observations["disk_angular_velocity"][2]


def get_distance_to_target(env_state: MJXEnvState) -> jnp.ndarray:
    return env_state.observations["xy_distance_to_target"]


def get_direction_to_target(env_state: MJXEnvState) -> jnp.ndarray:
    # is only available in XY plane -> calculate angle
    x, y = env_state.observations["unit_xy_direction_to_target"]
    # 
    return jnp.array([ x, y, env_state.observations["disk_rotation"][2] ])
    
    # NO key "direction_to_target" in observations
    x, y = env_state.observations["direction_to_target"]
    return jnp.arctan2(y, x)


def get_joint_positions(env_state: MJXEnvState, arm: int) -> jnp.ndarray:
    joint_positions = env_state.observations["joint_position"]

    num_joints = len(joint_positions) // 2
    return joint_positions[arm * num_joints : (arm + 1) * num_joints]


def get_joint_velocities(env_state: MJXEnvState, arm: int) -> jnp.ndarray:
    joint_velocities = env_state.observations["joint_velocity"]

    num_joints = len(joint_velocities) // 2
    return joint_velocities[arm * num_joints : (arm + 1) * num_joints]


# torque in newton meter
def get_joint_torques(env_state: MJXEnvState, arm: int) -> jnp.ndarray:
    joint_torques = env_state.observations["joint_actuator_force"]

    num_joints = len(joint_torques) // 2
    return joint_torques[arm * num_joints : (arm + 1) * num_joints]


# force in newton
def get_actuator_force(env_state: MJXEnvState, arm: int) -> jnp.ndarray:
    actuator_force = env_state.observations["actuator_force"]

    num_joints = len(actuator_force) // 2
    return actuator_force[arm * num_joints : (arm + 1) * num_joints]
