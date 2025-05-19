import jax.numpy as jnp
import numpy as np  
from tensorneat.common import State
from tensorneat.pipeline import Pipeline
import jax
import time
from NEAT.neat_controller import calculate_network_size_limits


### CODE FOR EXTENDING AN EXISTING GENOME INSTANCE ###


def add_segment_to_genome(
    genome: tuple[jnp.ndarray, jnp.ndarray], current_segment_count: int, arm_count=5
) -> tuple[jnp.ndarray, jnp.ndarray]:
    nodes, connections = genome

    nodes = np.array(nodes)
    connections = np.array(connections)
    seg_count = current_segment_count + 1

    # [!!!] don't forget to change this if we were to update the observation space
    input_size = seg_count * 2 * arm_count + 1
    output_size = seg_count * 2 * arm_count
    old_input_size = current_segment_count * 2 * arm_count + 1

    max_nodes, max_conn = calculate_network_size_limits(input_size, output_size)
    new_nodes = np.full((max_nodes, 4), np.nan)
    idx = 0
    n_idx = 0
    new_node_mapping = {}

    new_input_nodes = []
    new_output_nodes = []

    for arm in range(arm_count):
        for i in range(current_segment_count * 2):
            n = nodes[n_idx]
            n_idx += 1
            new_nodes[idx] = np.array([idx, n[1], n[2], n[3]])
            new_node_mapping[n[0]] = idx
            idx += 1


        attributes = [
            idx,
            np.random.normal(),
            0,
            np.random.randint(0, 3),
        ]

        new_nodes[idx] = np.array(attributes)
        new_input_nodes.append(idx)
        idx += 1

        attributes = [
            idx,  # Node ID
            np.random.normal(),  # Bias
            0,  # Node aggregation function
            np.random.randint(0, 3),  # Node activation function
        ]

        new_nodes[idx] = np.array(attributes)
        new_input_nodes.append(idx)
        idx += 1

    for i in range(n_idx, old_input_size):
        n = nodes[n_idx]
        n_idx += 1
        new_nodes[idx] = np.array([idx, n[1], n[2], n[3]])
        new_node_mapping[n[0]] = idx
        idx += 1

    output_start_idx = idx

    for arm in range(arm_count):
        for i in range(current_segment_count * 2):
            n = nodes[n_idx]
            n_idx += 1
            new_nodes[idx] = np.array([idx, n[1], n[2], n[3]])
            new_node_mapping[n[0]] = idx
            idx += 1


        attributes = [
            idx,
            np.random.normal(),
            0,
            np.random.randint(0, 3),
        ] 

        new_nodes[idx] = np.array(attributes)
        new_output_nodes.append(idx)
        idx += 1

        attributes = [idx, np.random.normal(), 0, np.random.randint(0, 3)]

        new_nodes[idx] = np.array(attributes)
        new_output_nodes.append(idx)
        idx += 1

    hidden_start_idx = idx + 1  # +1 for the padding between nodes and hidden nodes

    # add hidden nodes
    idx = hidden_start_idx
    for i in range(n_idx, len(nodes)):
        n = nodes[n_idx]
        n_idx += 1
        if np.isnan(n[0]):
            continue
        new_nodes[idx] = np.array([idx, n[1], n[2], n[3]])
        new_node_mapping[n[0]] = idx
        idx += 1

    new_connections = np.full((max_conn, 3), np.nan)

    # Add existing connections
    conn_idx = 0
    for c in connections:
        if np.isnan(c[0]):
            continue

        start_conn = new_node_mapping.get(c[0], c[0])
        end_conn = new_node_mapping.get(c[1], c[1])

        new_connections[conn_idx] = np.array([start_conn, end_conn, c[2]])
        conn_idx += 1

    np.random.seed(int(time.time()))

    # Add new connections to new input/output nodes

    for input_node in new_input_nodes:
        for output_node in range(output_start_idx, hidden_start_idx - 1):
            if np.random.random() < 0.5:
                weight = np.random.uniform(-1.0, 1.0)  # Random weight
                new_connections[conn_idx] = np.array([input_node, output_node, weight])
                conn_idx += 1

        for hidden_node in range(hidden_start_idx, idx):
            if np.random.random() < 0.3:
                weight = np.random.uniform(-1.0, 1.0)  # Random weight
                new_connections[conn_idx] = np.array([input_node, hidden_node, weight])
                conn_idx += 1

    for output_node in new_output_nodes:
        for input_node in range(0, output_start_idx):
            if np.random.random() < 0.5:
                weight = np.random.uniform(-1.0, 1.0)  # Random weight
                new_connections[conn_idx] = np.array([input_node, output_node, weight])
                conn_idx += 1

        for hidden_node in range(hidden_start_idx, idx):
            if np.random.random() < 0.3:
                weight = np.random.uniform(-1.0, 1.0)  # Random weight
                new_connections[conn_idx] = np.array([hidden_node, output_node, weight])
                conn_idx += 1

    return (
        jax.numpy.array(new_nodes, dtype=jax.numpy.float32),
        jax.numpy.array(new_connections, dtype=jax.numpy.float32),
    )


def extend_genome(
    state: State,
    pipeline: Pipeline,
    genomes: list[tuple[jnp.ndarray, jnp.ndarray]],
    current_segment_count: int,
    extra_segments: int = 0,
    arm_count: int = 5,
) -> State:
    """Extend the existing genome with additional inputs and outputs.
    This function creates a new genome with the same structure as the original one,
    but with additional nodes and connections to accommodate the extra inputs and outputs.
    The new genome is then transformed into a new state object.

    Args:
        state (State): State of the NEAT algorithm. (pass to the pipeline)
        pipeline (Pipeline): The NEAT pipeline.
        current_segment_count (int): The current segment count of the genome.
        extra_segments (int): The number of segments to add to the genome.
        arm_count (int): The number of arms in the genome.

    Returns:
        State: The new state object with the extended genome.
    """
    pop = pipeline.algorithm.ask(state)  # Retrieve population

    all_nodes = pop[0]
    all_conns = pop[1]

    for _ in range(extra_segments):
        for i in range(all_nodes.shape[0]):
            #print(len(genomes),len(genomes[i%len(genomes)]), len(genomes[i%len(genomes)][0]))
            current_genome = genomes[i%len(genomes)]

            #print(f"Genome {i%len(genomes)} selected for extension")

            new_nodes, new_conns = add_segment_to_genome(
                current_genome, current_segment_count, arm_count=arm_count
            )
            all_nodes = all_nodes.at[i].set(new_nodes)
            all_conns = all_conns.at[i].set(new_conns)

    # Update the state with the new population
    state = state.update(pop_nodes=all_nodes)
    state = state.update(pop_conns=all_conns)

    return state
