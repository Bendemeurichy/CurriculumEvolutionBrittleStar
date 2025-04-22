import jax.numpy as jnp
import numpy as np  # For random number generation
from tensorneat.common import State
from tensorneat.genome.utils import add_conn, add_node
from tensorneat.pipeline import Pipeline
import jax.random
import time
from NEAT.neat_controller import get_max_networks_dims


### CODE FOR EXTENDING AN EXISTING GENOME INSTANCE ###
def get_segment_count(connections: np.ndarray,arms=5) -> int:
    conn_count = {}
    for conn in connections:
        if np.isnan(conn[0]):
            continue
        if conn[1] not in conn_count:
            conn_count[conn[1]] = 0
        if conn[0] not in conn_count:
            conn_count[conn[0]] = 0
        conn_count[conn[1]] += 1
    input_size = len([i for i in conn_count if conn_count[i] == 0])

    return int((input_size - 1) / (2*arms))

def add_segment_to_genome(genome:tuple[jnp.ndarray, jnp.ndarray],current_segment_count:int, arm_count=5) -> tuple[jnp.ndarray, jnp.ndarray]:
    nodes,connections = genome
    
    # convert jax array to numpy array
    nodes = np.array(nodes)
    connections = np.array(connections)
    seg_count = current_segment_count + 1
  
    # [!!!] don't forget to change this if we were to update the observation space
    input_size = seg_count*2 * arm_count + 1
    output_size = seg_count*2 * arm_count
    old_input_size = current_segment_count*2 * arm_count + 1

    max_nodes, max_conn = get_max_networks_dims(input_size,  output_size)
    new_nodes = np.full((max_nodes, 4), np.nan)
    idx = 0
    n_idx = 0
    new_node_mapping = {}
    
    # Track newly added input and output nodes
    new_input_nodes = []
    new_output_nodes = []

    for arm in range(arm_count):
        for i in range(current_segment_count*2):
            n = nodes[n_idx]
            n_idx += 1
            new_nodes[idx] =  np.array([idx,n[1], n[2], n[3]])
            new_node_mapping[n[0]] = idx
            idx += 1
        
        # add the new input nodes and track them
        new_nodes[idx] = np.array([idx, 0.0, 0, 0])
        new_input_nodes.append(idx)
        idx += 1
        
        new_nodes[idx] = np.array([idx, 0.0, 0, 0])
        new_input_nodes.append(idx)
        idx += 1

    # Add the last input node
    for i in range(n_idx, old_input_size):
        n = nodes[n_idx]
        n_idx += 1
        new_nodes[idx] = np.array([idx, n[1], n[2], n[3]])
        new_node_mapping[n[0]] = idx
        idx += 1

    # Remember the output start index
    output_start_idx = idx

    # add output nodes
    for arm in range(arm_count):
        for i in range(current_segment_count*2):
            n = nodes[n_idx]
            n_idx += 1
            new_nodes[idx] = np.array([idx, n[1], n[2], n[3]])
            new_node_mapping[n[0]] = idx
            idx += 1
            
        # add the new output nodes and track them
        new_nodes[idx] = np.array([idx, 0.0, 0, 0])
        new_output_nodes.append(idx)
        idx += 1
        
        new_nodes[idx] = np.array([idx, 0.0, 0, 0])
        new_output_nodes.append(idx)
        idx += 1

    # Calculate hidden layer start index
    hidden_start_idx = idx + 1  # +1 for the padding between nodes and hidden nodes

    # add hidden nodes
    idx = hidden_start_idx
    for i in range(n_idx, len(nodes)):
        n = nodes[n_idx]
        n_idx += 1
        if np.isnan(n[0]):
            continue
        new_nodes[idx] = np.array([idx, n[1], n[2], n[3]])
        new_node_mapping[n[0]] = n[0]
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

    # Generate a seed for random number generation
    np.random.seed(int(time.time()))
    
    # Add new connections to new input/output nodes
    
    # Connect each new input node to a subset of hidden nodes and output nodes
    for input_node in new_input_nodes:
        # Connect to some output nodes
        for output_node in range(output_start_idx, hidden_start_idx - 1):
            # Add connections with 50% probability
            if np.random.random() < 0.5:
                weight = np.random.uniform(-1.0, 1.0)  # Random weight
                new_connections[conn_idx] = np.array([input_node, output_node, weight])
                conn_idx += 1
        
        # Connect to some hidden nodes if they exist
        for hidden_node in range(hidden_start_idx, idx):
            # Add connections with 30% probability
            if np.random.random() < 0.3:
                weight = np.random.uniform(-1.0, 1.0)  # Random weight
                new_connections[conn_idx] = np.array([input_node, hidden_node, weight])
                conn_idx += 1

    # Initialize connections for new output nodes
    # Connect each new output node from a subset of input nodes and hidden nodes
    for output_node in new_output_nodes:
        # Connect from some input nodes
        for input_node in range(0, output_start_idx):
            # Add connections with 50% probability
            if np.random.random() < 0.5:
                weight = np.random.uniform(-1.0, 1.0)  # Random weight
                new_connections[conn_idx] = np.array([input_node, output_node, weight])
                conn_idx += 1
        
        # Connect from some hidden nodes if they exist
        for hidden_node in range(hidden_start_idx, idx):
            # Add connections with 30% probability
            if np.random.random() < 0.3:
                weight = np.random.uniform(-1.0, 1.0)  # Random weight
                new_connections[conn_idx] = np.array([hidden_node, output_node, weight])
                conn_idx += 1
    
    return (jax.numpy.array(new_nodes, dtype=jax.numpy.float32), jax.numpy.array(new_connections, dtype=jax.numpy.float32))


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
