

from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.specification import BrittleStarMorphologySpecification
from render import visualize_mjcf


def create_morphology(
        morphology_specification: BrittleStarMorphologySpecification
        ) -> MJCFBrittleStarMorphology:
    morphology = MJCFBrittleStarMorphology(
            specification=morphology_specification
            )
    return morphology





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