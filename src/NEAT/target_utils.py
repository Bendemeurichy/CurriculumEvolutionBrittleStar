import jax.numpy as jnp
import jax

def get_distance_to_closest_target(disk_position, targets):
    """Calculate distance to closest target."""
    distances = jnp.array([
        jnp.linalg.norm(disk_position - targets[0][:2]),
        jnp.linalg.norm(disk_position - targets[1][:2])
    ])
    return jnp.min(distances), jnp.argmin(distances)

def get_direction_to_closest_target(disk_position, disk_rotation, targets):
    """Calculate direction to closest target relative to disk orientation."""
    # Get distances to both targets
    distance, closest_target_idx = get_distance_to_closest_target(disk_position, targets)
    
    # Get vector to target (assuming disk_position and target are in x,y coordinates)
    target_position = targets[closest_target_idx][:2]
    direction_vector = target_position - disk_position
    
    # Normalize the direction vector
    distance = jnp.linalg.norm(direction_vector)
    unit_direction = direction_vector / jnp.where(distance > 0, distance, 1.0)
    
    # Calculate angle to target in world coordinates
    target_angle = jnp.arctan2(unit_direction[1], unit_direction[0])
    
    # Calculate angle difference (between -pi and pi)
    # This gives relative angle between disk orientation and target
    angle_diff = jnp.mod(target_angle - disk_rotation + jnp.pi, 2 * jnp.pi) - jnp.pi
    
    return jnp.array([angle_diff]), closest_target_idx
