import jax.numpy as jnp
import numpy as np  # For random number generation
from tensorneat.common import State
from tensorneat.genome.utils import add_conn, add_node
from tensorneat.pipeline import Pipeline
import jax.random
import time


### CODE FOR EXTENDING AN EXISTING GENOME INSTANCE ###


def extend_genome(
    state: State,
    pipeline: Pipeline,
    genome: tuple[jnp.ndarray, jnp.ndarray],
    extra_inputs: int = 0,
    extra_outputs: int = 0,
) -> tuple[Pipeline, State]:
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
        pipeline,State: The new state object with the extended genome.
    """
    old_output_start = pipeline.algorithm.genome.output_idx[0] - extra_inputs
    new_output_start = old_output_start + extra_inputs
    output_amount = len(pipeline.algorithm.genome.output_idx)

    nodes, conns = shift_index_outputs(
        genome[0], genome[1], old_output_start, new_output_start
    )

    pop = pipeline.algorithm.ask(state)  # Retrieve population
    all_nodes = pop[0]
    all_conns = pop[1]

    seed = int(time.time() * 1000)
    key = jax.random.key(seed)

    for instance in range(len(all_nodes)):
        all_nodes[instance] = nodes
        for i in range(extra_inputs, 0, -1):
            new_idx = new_output_start - i
            key, subkey = jax.random.split(key)
            bias = jax.random.normal(subkey, shape=()) * 0.5
            key, subkey = jax.random.split(key)
            activation = jax.random.randint(subkey, shape=(), minval=0, maxval=3)
            attributes = [
                new_idx,  # Node ID
                bias,  # bias
                0,  # aggregation function
                activation,  # activation
            ]
            all_nodes[instance] = add_node(all_nodes[instance], attributes)
        for i in range(extra_outputs):
            new_idx = new_output_start + output_amount + i
            key, subkey = jax.random.split(key)
            bias = jax.random.normal(subkey, shape=()) * 0.5
            key, subkey = jax.random.split(key)
            activation = jax.random.randint(subkey, shape=(), minval=0, maxval=3)
            attributes = [
                new_idx,  # Node ID
                bias,  # bias
                0,  # aggregation function
                activation,  # activation
            ]
            all_nodes[instance] = add_node(all_nodes[instance], attributes)

    for instance in range(len(all_conns)):
        all_conns[instance] = conns

        # new inputs to all non-input nodes
        for i in range(
            pipeline.algorithm.genome.input_idx[-1] - extra_inputs,
            pipeline.algorithm.genome.input_idx[-1] + 1,
        ):
            for j in range(all_nodes[0].shape[0]):
                if j not in pipeline.algorithm.genome.input_idx and ~jnp.isnan(
                    all_nodes[0][j, 0]
                ):
                    key, subkey = jax.random.split(key)
                    attributes = [
                        i,  # From node ID
                        j,  # To node ID
                        jax.random.uniform(subkey, minval=0.1, maxval=1.0),  # weight
                    ]
                    all_conns[instance] = add_conn(all_conns[instance], attributes)

        # new outputs from all other nodes, except the new inputs
        for i in range(extra_outputs):
            output_idx = new_output_start + output_amount + i
            for j in range(all_nodes[0].shape[0]):
                # Skip if node doesn't exist, is a new output, or is a new input
                if (
                    jnp.isnan(all_nodes[0][j, 0])  # Node doesn't exist
                    or j >= new_output_start + output_amount  # Is a new output
                    or (
                        new_output_start - extra_inputs <= j < new_output_start
                    )  # Is a new input
                ):
                    continue

                key, subkey = jax.random.split(key)
                attributes = [
                    j,  # From node ID
                    output_idx,  # To node ID
                    jax.random.uniform(subkey, minval=0.1, maxval=1.0),  # weight
                ]
                all_conns[instance] = add_conn(all_conns[instance], attributes)

    pop = all_nodes, all_conns
    state = pipeline.algorithm.tell(state, pop)  # Update state with population

    return pipeline, state


def shift_index_outputs(
    nodes: jnp.ndarray,
    connections: jnp.ndarray,
    old_output_start: int,
    new_output_start: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shift the indices of nodes and connections in the genome to accommodate new inputs and outputs.
    Use JAX style programming to optimize the process.

    Args:
        nodes jnp.ndarray: nodes that need to be shifted
        connections jnp.ndarray: connections that need to be shifted
        old_output_start int: index of the first output node
        new_output_start int: index of the first new output node

    Returns:
        tuple: shifted nodes and connections
    """
    diff = new_output_start - old_output_start

    # Create masks for which elements to update
    node_mask = (nodes[:, 0] >= old_output_start) & ~jnp.isnan(nodes[:, 0])
    conn_from_mask = (connections[:, 0] >= old_output_start) & ~jnp.isnan(
        connections[:, 0]
    )
    conn_to_mask = (connections[:, 1] >= old_output_start) & ~jnp.isnan(
        connections[:, 1]
    )

    # Apply shifts to arrays where masks are True
    new_nodes = nodes.at[:, 0].add(diff * node_mask)
    new_connections = connections.at[:, 0].add(diff * conn_from_mask)
    new_connections = new_connections.at[:, 1].add(diff * conn_to_mask)

    return new_nodes, new_connections
