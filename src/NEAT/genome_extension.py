import copy
import jax.numpy as jnp
import numpy as np  # For random number generation
from tensorneat.common import State
from tensorneat.genome.utils import add_conn, add_node
from tensorneat.pipeline import Pipeline
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from NEAT.neat_controller import init_pipeline
from NEAT.neat_problem import BrittleStarEnv
from NEAT.visualize import load_model


### CODE FOR EXTENDING AN EXISTING GENOME INSTANCE ###


def extend_genome(
    state: State,
    pipeline: Pipeline,
    genome: tuple[jnp.ndarray, jnp.ndarray],
    extra_inputs: int = 0,
    extra_outputs: int = 0,
) -> State:
    """Extend the existing genome with additional inputs and outputs.
    This function creates a new genome with the same structure as the original one,
    but with additional nodes and connections to accommodate the extra inputs and outputs.
    The new genome is then transformed into a new state object.

    Args:
        state (State): State of the NEAT algorithm. (pass to the pipeline)
        genome (tuple[jnp.ndarray, jnp.ndarray]): The result of a previous NEAT run. (nodes, conns)
        extra_inputs (int, optional): The amount of extra inputs wanted. Defaults to 0.
        extra_outputs (int, optional): The amount of extra outputs wanted. Defaults to 0.

    Returns:
        State: The new state object with the extended genome.
    """

    nodes, conns = copy.deepcopy(genome)  # Unpack the genome tuple

    pop = pipeline.algorithm.ask(state)  # Retrieve population
    all_nodes = pop[0]
    all_conns = pop[1]

    # Reconstruct input_idx and output_idx of smaller genome
    input_idx = pipeline.algorithm.genome.input_idx
    output_idx = pipeline.algorithm.genome.output_idx
    input_idx = [i for i in range(len(input_idx) - extra_inputs)]
    output_idx = [i for i in range(len(output_idx) - extra_outputs)]

    new_input_idx = set(input_idx)
    new_output_idx = set(output_idx)

    for instance in range(all_nodes.shape[0]):
        all_nodes[instance] = nodes
        all_conns[instance] = conns
        # Add extra inputs and outputs to the genome
        for _ in range(extra_inputs):
            new_idx = all_nodes[instance].shape[0]

            attributes = [
                new_idx,  # Node ID
                np.random.normal(0, 0.5),  # bias
                0,  # aggregation function
                np.random.randint(0, 3),  # activation (random from 0 to 2)
            ]
            all_nodes[instance] = add_node(all_nodes[instance], attributes)
            new_input_idx.add(new_idx)

        for _ in range(extra_outputs):
            new_idx = all_nodes[instance].shape[0]

            attributes = [
                new_idx,  # Node ID
                np.random.normal(0, 0.5),  # bias
                0,  # aggregation function
                np.random.randint(0, 3),  # activation (random from 0 to 2)
            ]
            all_nodes[instance] = add_node(all_nodes[instance], attributes)
            new_output_idx.add(new_idx)

        # add connections from the new inputs to all the not input nodes
        for i in new_input_idx:
            for j in range(all_nodes[instance].shape[0]):
                if (
                    all_nodes[instance][j, 0] not in new_input_idx
                    and all_nodes[instance][j, 0]
                    not in pipeline.algorithm.genome.input_idx
                ):
                    attributes = [
                        i,  # input node
                        all_nodes[instance][j, 0],  # output node
                        np.random.uniform(-1, 1),  # weight
                    ]
                    all_conns[instance] = add_conn(all_conns[instance], attributes)

        # add connections from all nodes to the new outputs
        for i in new_output_idx:
            for j in range(all_nodes[instance].shape[0]):
                if (
                    all_nodes[instance][j, 0] not in new_output_idx
                    and all_nodes[instance][j, 0]
                    not in pipeline.algorithm.genome.output_idx
                ):
                    attributes = [
                        all_nodes[instance][j, 0],  # input node
                        i,  # output node
                        np.random.uniform(-1, 1),  # weight
                    ]
                    all_conns[instance] = add_conn(all_conns[instance], attributes)

    # Weird hack because pipeline is already configured for larger genome
    pipeline.algorithm.genome.input_idx = jnp.array(list(new_input_idx))
    pipeline.algorithm.genome.output_idx = jnp.array(list(new_output_idx))

    pop = all_nodes, all_conns
    state = pipeline.algorithm.tell(state, pop)  # Update state with population

    return pipeline, state
