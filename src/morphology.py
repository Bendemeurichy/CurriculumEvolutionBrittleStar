

from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification,default_joint_specification,linear_interpolation
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.specification import BrittleStarMorphologySpecification,BrittleStarDiskSpecification,BrittleStarActuationSpecification, BrittleStarSensorSpecification, BrittleStarArmSegmentSpecification,BrittleStarArmSpecification
from render import visualize_mjcf
from typing import List, Union

import numpy as np
import NEAT.config as config
def create_morphology(
        morphology_specification: BrittleStarMorphologySpecification
        ) -> MJCFBrittleStarMorphology:
    morphology = MJCFBrittleStarMorphology(
            specification=morphology_specification
            )
    return morphology



def default_arm_segment_specification(
    alpha: float,
) -> BrittleStarArmSegmentSpecification:
    in_plane_joint_specification = default_joint_specification(
        range=60 / 180 * np.pi
    )  # 30
    out_of_plane_joint_specification = default_joint_specification(
        range=45 / 180 * np.pi
    )  # 5

    radius = linear_interpolation(
        alpha=alpha, start=config.START_SEGMENT_RADIUS, stop=config.STOP_SEGMENT_RADIUS
    )
    length = linear_interpolation(
        alpha=alpha, start=config.START_SEGMENT_LENGTH, stop=config.STOP_SEGMENT_LENGTH
    )

    segment_specification = BrittleStarArmSegmentSpecification(
        radius=radius,
        length=length,
        in_plane_joint_specification=in_plane_joint_specification,
        out_of_plane_joint_specification=out_of_plane_joint_specification,
    )
    return segment_specification


def default_arm_specification(num_segments_per_arm: int) -> BrittleStarArmSpecification:
    segment_specifications = list()
    for segment_index in range(num_segments_per_arm):
        segment_specification = default_arm_segment_specification(
            alpha=segment_index / num_segments_per_arm
        )
        segment_specifications.append(segment_specification)

    arm_specification = BrittleStarArmSpecification(
        segment_specifications=segment_specifications
    )
    return arm_specification



def default_brittle_star_morphology_specification(
    num_arms: int = 5,
    num_segments_per_arm: Union[int, List[int]] = 5,
    use_tendons: bool = False,
    use_p_control: bool = False,
    use_torque_control: bool = False,
    radius_to_strength_factor: float = 200,
    num_contact_sensors_per_segment: int = 1,
    diameter: float = config.DISK_DIAMETER,
    height: float = config.DISK_HEIGHT,
) -> BrittleStarMorphologySpecification:
    disk_specification = BrittleStarDiskSpecification(
        diameter=diameter, height=height
    )

    if isinstance(num_segments_per_arm, int):
        num_segments_per_arm = [num_segments_per_arm] * num_arms
    else:
        assert len(num_segments_per_arm) == num_arms, (
            f"Length of the 'num_segments_per_arm' input must be"
            f"equal to the 'num_arms' input."
        )

    arm_specifications = list()
    for num_segments in num_segments_per_arm:
        arm_specification = default_arm_specification(num_segments_per_arm=num_segments)
        arm_specifications.append(arm_specification)

    actuation_specification = BrittleStarActuationSpecification(
        use_tendons=use_tendons,
        use_p_control=use_p_control,
        use_torque_control=use_torque_control,
        radius_to_strength_factor=radius_to_strength_factor,
    )
    sensor_specification = BrittleStarSensorSpecification(
        num_contact_sensors_per_segment=num_contact_sensors_per_segment
    )

    specification = BrittleStarMorphologySpecification(
        disk_specification=disk_specification,
        arm_specifications=arm_specifications,
        actuation_specification=actuation_specification,
        sensor_specification=sensor_specification,
    )

    return specification


if __name__ == "__main__":
    morphology_specification = default_brittle_star_morphology_specification(
            num_arms=5, 
            num_segments_per_arm=4, 
            # Whether or not to use position-based control (i.e. the actuation or control inputs are target joint positions).
            use_p_control=True,
            # Whether or not to use torque-based control (i.e. the actuation or control inputs are target joint torques).
            use_torque_control=False
            )
    morphology = create_morphology(morphology_specification=morphology_specification)
    visualize_mjcf(mjcf=morphology)

    morphology_specification = default_brittle_star_morphology_specification(
            num_arms=5, num_segments_per_arm=[1, 2, 3, 4, 5], use_p_control=True, use_torque_control=False
            )
    morphology = create_morphology(morphology_specification=morphology_specification)
    visualize_mjcf(mjcf=morphology)