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
    new_input_idx = set()
    new_output_idx = set()

    # TODO: Modify the genome to add extra inputs and outputs
    for instance in range(all_nodes.shape[0]):
        all_nodes[instance] = nodes
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

    pipeline.algorithm.genome.input_idx = jnp.concatenate(
        [pipeline.algorithm.genome.input_idx, jnp.array(list(new_input_idx))]
    )
    pipeline.algorithm.genome.output_idx = jnp.concatenate(
        [pipeline.algorithm.genome.output_idx, jnp.array(list(new_output_idx))]
    )
    pipeline.algorithm.genome.num_inputs += extra_inputs
    pipeline.algorithm.genome.num_outputs += extra_outputs

    pop = all_nodes, all_conns
    state = pipeline.algorithm.tell(state, pop)  # Update state with population

    return pipeline, state


problem = BrittleStarEnv()  # Use the new RLEnv-based class
# Get input and output dimensions
num_inputs, num_outputs = problem._input_dims, problem._output_dims
print(f"Environment requires {num_inputs} inputs and {num_outputs} outputs")

pipeline = init_pipeline(problem)

print("Initializing TensorNEAT state...")
state = pipeline.setup()

genome = load_model("./models/best_genome.pkl")


print(f"shape of nodes: {genome[0].shape}")
print(f"shape of conns: {genome[1].shape}")

# Print the nodes and connections arrays
nodes, conns = genome


nodes = jnp.array(nodes)
conns = jnp.array(conns)

print("representation?")
print(pipeline.algorithm.genome.repr(state, nodes, conns))

# drop nans
nodes = nodes[~jnp.isnan(nodes).any(axis=1)]
conns = conns[~jnp.isnan(conns).any(axis=1)]


print("Nodes:\n", nodes)
print("Connections:\n", conns)


# count number of nodes with 4th value == 0 , 1 and 2
num_inputs = jnp.sum(nodes[:, 3] == 0)
num_outputs = jnp.sum(nodes[:, 3] == 2)
num_hidden = jnp.sum(nodes[:, 3] == 1)
print("Number of inputs:", num_inputs)
print("Number of outputs:", num_outputs)
print("Number of hidden nodes:", num_hidden)
print(pipeline.algorithm.genome.input_idx)
