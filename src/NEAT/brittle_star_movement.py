import jax
import jax.numpy as jnp
from tensorneat.problem import BaseProblem

from observations import get_joint_positions, get_distance_to_target


class MovementToTargetProblem(BaseProblem):
    jitable = True

    def __init__(self, max_steps=100):
        """
        max_steps: number of simulation steps per evaluation.
        """
        self.max_steps = max_steps

    @property
    def input_shape(self):
        # For two arms with 2 inputs each (relative x and y),
        # we expect a flattened input of size 4.
        return (4,)

    @property
    def output_shape(self):
        # Two arms with 2 outputs each (control updates) yield 4 outputs.
        return (4,)

    def evaluate(self, state, randkey, act_func, params):
        """
        Simulate movement toward target.
        At each timestep an observation is constructed as the relative vector from
        current position to target, duplicated for the two arms.
        The network outputs 4 values; we take the average of the two pairs as the
        delta update for position.
        Fitness is defined by the imported fitness function.
        """
        pass

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        """
        Run one evaluation and print details including the trajectory.
        """
        pass
