"""Module for extracting observations from the brittle star environment."""
import jax.numpy as jnp
from moojoco.environment.mjx_env import MJXEnvState
import jax


def extract_disk_position(env_state: MJXEnvState) -> jnp.ndarray:
    """Extract the 2D position of the brittle star's central disk.
    
    Args:
        env_state: Current environment state
        
    Returns:
        2D position vector [x, y]
    """
    return env_state.observations["disk_position"][:2]


def extract_disk_direction(env_state: MJXEnvState) -> jnp.ndarray:
    """Extract the rotation angle of the brittle star's central disk.
    
    Args:
        env_state: Current environment state
        
    Returns:
        Rotation angle around z-axis
    """
    return env_state.observations["disk_rotation"][2]


def extract_disk_velocity(env_state: MJXEnvState) -> jnp.ndarray:
    """Extract the 2D velocity of the brittle star's central disk.
    
    Args:
        env_state: Current environment state
        
    Returns:
        2D velocity vector [vx, vy]
    """
    return env_state.observations["disk_linear_velocity"][:2]


def extract_disk_angular_velocity(env_state: MJXEnvState) -> jnp.ndarray:
    """Extract the angular velocity of the brittle star's central disk.
    
    Args:
        env_state: Current environment state
        
    Returns:
        Angular velocity around z-axis
    """
    return env_state.observations["disk_angular_velocity"][2]


def calculate_distance_to_target(env_state: MJXEnvState) -> jnp.ndarray:
    """Calculate the distance to the target.
    
    Args:
        env_state: Current environment state
        
    Returns:
        Distance to target
    """
    return env_state.observations["xy_distance_to_target"]


def calculate_direction_to_target(env_state: MJXEnvState) -> jnp.ndarray:
    """Calculate the relative direction to the target from the brittle star's perspective.
    
    Args:
        env_state: Current environment state
        
    Returns:
        Relative angle to target in radians
    """
    # Get unit vector pointing to target
    target_direction_x, target_direction_y = env_state.observations["unit_xy_direction_to_target"]
    
    # Get disk's facing direction
    disk_angle = env_state.observations["disk_rotation"][2]
    
    # Calculate the target angle in world space
    target_angle = jnp.arctan2(target_direction_y, target_direction_x)
    
    # Calculate the angle difference (between -pi and pi)
    # This gives the relative angle between facing direction and target direction
    angle_diff = jnp.mod(target_angle - disk_angle + jnp.pi, 2 * jnp.pi) - jnp.pi
    
    return jnp.array([angle_diff])


def extract_joint_positions(env_state: MJXEnvState, arm: int) -> jnp.ndarray:
    """Extract the joint positions for a specific arm.
    
    Args:
        env_state: Current environment state
        arm: Arm index
        
    Returns:
        Joint positions for the specified arm
    """
    joint_positions = env_state.observations["joint_position"]
    num_joints_per_arm = len(joint_positions) // 5  # 5 arms
    return joint_positions[arm * num_joints_per_arm : (arm + 1) * num_joints_per_arm]


def extract_joint_velocities(env_state: MJXEnvState, arm: int) -> jnp.ndarray:
    """Extract the joint velocities for a specific arm.
    
    Args:
        env_state: Current environment state
        arm: Arm index
        
    Returns:
        Joint velocities for the specified arm
    """
    joint_velocities = env_state.observations["joint_velocity"]
    num_joints_per_arm = len(joint_velocities) // 5
    return joint_velocities[arm * num_joints_per_arm : (arm + 1) * num_joints_per_arm]


def extract_joint_torques(env_state: MJXEnvState, arm: int) -> jnp.ndarray:
    """Extract the joint torques for a specific arm.
    
    Args:
        env_state: Current environment state
        arm: Arm index
        
    Returns:
        Joint torques for the specified arm in newton-meters
    """
    joint_torques = env_state.observations["joint_actuator_force"]
    num_joints_per_arm = len(joint_torques) // 5
    return joint_torques[arm * num_joints_per_arm : (arm + 1) * num_joints_per_arm]


def extract_actuator_force(env_state: MJXEnvState, arm: int) -> jnp.ndarray:
    """Extract the actuator forces for a specific arm.
    
    Args:
        env_state: Current environment state
        arm: Arm index
        
    Returns:
        Actuator forces for the specified arm in newtons
    """
    actuator_force = env_state.observations["actuator_force"]
    num_joints_per_arm = len(actuator_force) // 5
    return actuator_force[arm * num_joints_per_arm : (arm + 1) * num_joints_per_arm]


def extract_all_observations(env_state: MJXEnvState) -> dict:
    """Extract all observations from the environment state.
    
    Args:
        env_state: Current environment state
        
    Returns:
        Dictionary containing all available observations
    """
    return {
        "disk_position": extract_disk_position(env_state),
        "disk_direction": extract_disk_direction(env_state),
        "disk_velocity": extract_disk_velocity(env_state),
        "disk_angular_velocity": extract_disk_angular_velocity(env_state),
        "distance_to_target": calculate_distance_to_target(env_state),
        "direction_to_target": calculate_direction_to_target(env_state),
        "joint_positions": [extract_joint_positions(env_state, arm) for arm in range(5)],
        "joint_velocities": [extract_joint_velocities(env_state, arm) for arm in range(5)],
        "joint_torques": [extract_joint_torques(env_state, arm) for arm in range(5)],
        "actuator_force": [extract_actuator_force(env_state, arm) for arm in range(5)],
    }


# For backward compatibility
get_disk_position = extract_disk_position
get_disk_direction = extract_disk_direction
get_disk_velocity = extract_disk_velocity
get_disk_angular_velocity = extract_disk_angular_velocity
get_distance_to_target = calculate_distance_to_target
get_direction_to_target = calculate_direction_to_target
get_joint_positions = extract_joint_positions
get_joint_velocities = extract_joint_velocities
get_joint_torques = extract_joint_torques
get_actuator_force = extract_actuator_force
observe = extract_all_observations
