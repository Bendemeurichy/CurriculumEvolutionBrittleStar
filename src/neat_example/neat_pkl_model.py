import jax, jax.numpy as jnp
import os
import pickle
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.func_fit import CustomFuncFit
from tensorneat.common import ACT, AGG

def add_square_activation():
    def square(x):
        return x ** 2
    try:
        ACT.add_func("square", square) # Idk sometimes it's already added
    except ValueError:
        pass

def train_and_save_pkl(output_dir="./", pagie_polynomial=None, generation_limit=50):
    custom_problem = CustomFuncFit(
        func=pagie_polynomial,
        low_bounds=[-1, -1],
        upper_bounds=[1, 1],
        method="sample",
        num_samples=100,
    )

    add_square_activation()

    algorithm = NEAT(
        pop_size=10000,
        species_size=20,
        survival_threshold=0.01,
        genome=DefaultGenome(
            num_inputs=2,
            num_outputs=1,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=[ACT.identity, ACT.inv, ACT.square],
                aggregation_options=[AGG.sum, AGG.product],
            ),
            output_transform=ACT.identity,
        ),
    )

    pipeline = Pipeline(
        algorithm,
        problem=custom_problem,
        generation_limit=generation_limit,
        fitness_target=-1e-6,
        seed=42,
    )
    
    state = pipeline.setup()
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        state, best = pipeline.auto_run(state)
        print("Evolution completed successfully")
        
        pipeline.show(state, best)
        
        model_path = os.path.join(output_dir, "best_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(best, f)
        print(f"Best model saved to {model_path}")
        
        # Visualize the network
        network = algorithm.genome.network_dict(state, *best)
        algorithm.genome.visualize(network, save_path=os.path.join(output_dir, "network_pkl.svg"))
        print(f"NN visualization saved to {os.path.join(output_dir, 'network_pkl.svg')}")
        
        return model_path
    
    except Exception as e:
        print(f"Error during evolution: {str(e)}")
        return None

def load_and_use_pkl_model(model_path, inputs, pagie_polynomial=None):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded from pickle, length:", len(model))
    
    add_square_activation()
    
    algorithm = NEAT(
        pop_size=10000,
        species_size=20,
        survival_threshold=0.01,
        genome=DefaultGenome(
            num_inputs=2,
            num_outputs=1,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=[ACT.identity, ACT.inv, ACT.square],
                aggregation_options=[AGG.sum, AGG.product],
            ),
            output_transform=ACT.identity,
        ),
    )
    
    problem = CustomFuncFit(
        func=pagie_polynomial,
        low_bounds=[-1, -1],
        upper_bounds=[1, 1],
        method="sample",
        num_samples=100,
    )
    pipeline = Pipeline(
        algorithm,
        problem=problem,
        generation_limit=1,
        fitness_target=-1e-6,
        seed=42,
    )
    state = pipeline.setup()
    
    inputs_array = jnp.array(inputs)
    
    transformed_model = algorithm.transform(state, model)
    prediction = algorithm.forward(state, transformed_model, inputs_array)
    
    actual = pagie_polynomial(inputs_array)[0] if pagie_polynomial else None
    error = abs(prediction[0] - actual) if actual is not None else None
    
    return prediction, actual, error

def test_pkl_model(model_path, test_inputs, pagie_polynomial):
    print("\nTesting pickle-based model:")
    if os.path.exists(model_path):
        for test_input in test_inputs:
            prediction, actual, error = load_and_use_pkl_model(model_path, test_input, pagie_polynomial)
            print(f"Input {test_input}: Prediction {prediction}, Actual: {actual}, Error: {error}")
    else:
        print(f"No pickle model file found at {model_path}.")



if __name__ == "__main__":

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    

    pkl_model_path = os.path.join(output_dir, "best_model.pkl")
    
    def pagie_polynomial(inputs):
        x, y = inputs
        res = 1 / (1 + jnp.pow(x, -4)) + 1 / (1 + jnp.pow(y, -4))
        return jnp.array([res])
    
    pkl_model_path = train_and_save_pkl(output_dir, pagie_polynomial, generation_limit=10)

    test_inputs = [
        [0.5, 0.5],
        [0.1, 0.9],
        [-0.5, -0.5],
        [0.0, 0.0]
    ]
    test_pkl_model(pkl_model_path, test_inputs, pagie_polynomial)
    