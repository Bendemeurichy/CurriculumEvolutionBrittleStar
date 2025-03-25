# Prepartion
import jax
import jax.numpy as jnp
from tensorneat.problem import BaseProblem


class MovementProblem(BaseProblem):
    jitable = True

    def evaluate(self, state, randkey, act_func, params):
        # Use ``act_func(state, params, inputs)`` to do network forward

        # do batch forward for all inputs (using jax.vamp)
        predict = jax.vmap(act_func, in_axes=(None, None, 0))(
            state, params, INPUTS
        )  # should be shape (1000, 1)

        # calculate loss
        loss = jnp.mean(jnp.square(predict - LABELS))

        # return negative loss as fitness
        # TensorNEAT maximizes fitness, equivalent to minimizes loss
        return -loss

    @property
    def input_shape(self):
        # the input shape that the act_func expects
        return (2,)

    @property
    def output_shape(self):
        # the output shape that the act_func returns
        return (1,)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        # shocase the performance of one individual
        predict = jax.vmap(act_func, in_axes=(None, None, 0))(state, params, INPUTS)

        loss = jnp.mean(jnp.square(predict - LABELS))

        msg = ""
        for i in range(INPUTS.shape[0]):
            msg += f"input: {INPUTS[i]}, target: {LABELS[i]}, predict: {predict[i]}\n"
        msg += f"loss: {loss}\n"
        print(msg)
