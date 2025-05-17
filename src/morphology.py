"""Module for brittle star morphology creation and configuration."""

import numpy as np
from typing import List, Union

from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import (
    default_joint_specification,
    linear_interpolation,
)
from biorobot.brittle_star.mjcf.morphology.specification.specification import (
    BrittleStarMorphologySpecification,
    BrittleStarDiskSpecification,
    BrittleStarActuationSpecification,
    BrittleStarSensorSpecification,
    BrittleStarArmSegmentSpecification,
    BrittleStarArmSpecification,
)

import NEAT.config as config


def create_morphology(
    morphology_specification: BrittleStarMorphologySpecification
) -> MJCFBrittleStarMorphology:
    """Create a brittle star morphology from a specification.
    
    Args:
        morphology_specification: The specification for the morphology
        
    Returns:
        A brittle star morphology
    """
    return MJCFBrittleStarMorphology(specification=morphology_specification)


def create_arm_segment_specification(
    alpha: float,
    start_radius: float = config.START_SEGMENT_RADIUS,
    stop_radius: float = config.STOP_SEGMENT_RADIUS,
    start_length: float = config.START_SEGMENT_LENGTH,
    stop_length: float = config.STOP_SEGMENT_LENGTH,
    in_plane_angle_deg: float = 60.0,
    out_of_plane_angle_deg: float = 45.0,
) -> BrittleStarArmSegmentSpecification:
    """Create a specification for an arm segment.
    
    Args:
        alpha: Interpolation factor for segment properties (0 to 1)
        start_radius: Radius at the base
        stop_radius: Radius at the tip
        start_length: Length at the base
        stop_length: Length at the tip
        in_plane_angle_deg: In-plane joint angle in degrees
        out_of_plane_angle_deg: Out-of-plane joint angle in degrees
        
    Returns:
        An arm segment specification
    """
    in_plane_joint_specification = default_joint_specification(
        range=in_plane_angle_deg / 180 * np.pi
    )
    out_of_plane_joint_specification = default_joint_specification(
        range=out_of_plane_angle_deg / 180 * np.pi
    )

    radius = linear_interpolation(alpha=alpha, start=start_radius, stop=stop_radius)
    length = linear_interpolation(alpha=alpha, start=start_length, stop=stop_length)

    return BrittleStarArmSegmentSpecification(
        radius=radius,
        length=length,
        in_plane_joint_specification=in_plane_joint_specification,
        out_of_plane_joint_specification=out_of_plane_joint_specification,
    )


def create_arm_specification(num_segments: int) -> BrittleStarArmSpecification:
    """Create an arm specification with the given number of segments.
    
    Args:
        num_segments: Number of segments in the arm
        
    Returns:
        An arm specification
    """
    segment_specifications = []
    for segment_index in range(num_segments):
        alpha = segment_index / num_segments if num_segments > 1 else 0
        segment_specification = create_arm_segment_specification(alpha=alpha)
        segment_specifications.append(segment_specification)

    return BrittleStarArmSpecification(segment_specifications=segment_specifications)


def create_brittle_star_morphology_specification(
    num_arms: int = 5,
    num_segments_per_arm: Union[int, List[int]] = 5,
    use_tendons: bool = False,
    use_p_control: bool = False,
    use_torque_control: bool = False,
    radius_to_strength_factor: float = 200,
    num_contact_sensors_per_segment: int = 1,
    disk_diameter: float = config.DISK_DIAMETER,
    disk_height: float = config.DISK_HEIGHT,
) -> BrittleStarMorphologySpecification:
    """Create a complete brittle star morphology specification.
    
    Args:
        num_arms: Number of arms
        num_segments_per_arm: Number of segments per arm (int or list of ints)
        use_tendons: Whether to use tendons for control
        use_p_control: Whether to use position control
        use_torque_control: Whether to use torque control
        radius_to_strength_factor: Factor to convert radius to strength
        num_contact_sensors_per_segment: Number of contact sensors per segment
        disk_diameter: Diameter of the central disk
        disk_height: Height of the central disk
        
    Returns:
        A complete brittle star morphology specification
        
    Raises:
        ValueError: If num_segments_per_arm is a list with length != num_arms
    """
    disk_specification = BrittleStarDiskSpecification(
        diameter=disk_diameter, height=disk_height
    )

    if isinstance(num_segments_per_arm, int):
        num_segments_per_arm = [num_segments_per_arm] * num_arms
    elif len(num_segments_per_arm) != num_arms:
        raise ValueError(
            f"Length of num_segments_per_arm ({len(num_segments_per_arm)}) "
            f"must equal num_arms ({num_arms})"
        )

    arm_specifications = [
        create_arm_specification(num_segments=num_segments)
        for num_segments in num_segments_per_arm
    ]

    actuation_specification = BrittleStarActuationSpecification(
        use_tendons=use_tendons,
        use_p_control=use_p_control,
        use_torque_control=use_torque_control,
        radius_to_strength_factor=radius_to_strength_factor,
    )
    
    sensor_specification = BrittleStarSensorSpecification(
        num_contact_sensors_per_segment=num_contact_sensors_per_segment
    )

    return BrittleStarMorphologySpecification(
        disk_specification=disk_specification,
        arm_specifications=arm_specifications,
        actuation_specification=actuation_specification,
        sensor_specification=sensor_specification,
    )

