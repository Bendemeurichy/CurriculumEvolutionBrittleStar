import jax, jax.numpy as jnp
import os
import pickle
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.func_fit import CustomFuncFit
from tensorneat.common import ACT, AGG

def pagie_polynomial(inputs):
    x, y = inputs
    res = 1 / (1 + jnp.pow(x, -4)) + 1 / (1 + jnp.pow(y, -4))
    return jnp.array([res])


def train(checkpoint_dir="checkpoints", save_interval=2):
  # Create checkpoint directory if it doesn't exist
  os.makedirs(checkpoint_dir, exist_ok=True)
  
  custom_problem = CustomFuncFit(
      func=pagie_polynomial,
      low_bounds=[-1, -1],
      upper_bounds=[1, 1],
      method="sample",
      num_samples=100,
  )


  def square(x):
      return x ** 2
  ACT.add_func("square", square)

  algorithm = NEAT(
      pop_size=10000,
      species_size=20,
      survival_threshold=0.01,
      genome=DefaultGenome(
          num_inputs=2,
          num_outputs=1,
          init_hidden_layers=(),
          node_gene=BiasNode(
              # Using (identity, inversion, square) 
              # as possible activation functions
              activation_options=[ACT.identity, ACT.inv, ACT.square],
              # Using (sum, product) as possible aggregation functions
              aggregation_options=[AGG.sum, AGG.product],
          ),
          output_transform=ACT.identity,
      ),
  )


  pipeline = Pipeline(
      algorithm=algorithm,
      problem=custom_problem,
      generation_limit=10,
      fitness_target=-1e-4,
      seed=42,
  )

  # Initialize state
  state = pipeline.setup()
  
  # Run manually with checkpoints
  best = None
  for i in range(pipeline.generation_limit):
    state, step_best = pipeline.step(state)
    
    # Update best if needed
    if best is None or step_best.fitness > best.fitness:
        best = step_best
    
    # Save checkpoint at specified intervals
    if (i + 1) % save_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_gen_{i+1}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'state': state,
                'best': best,
                'generation': i+1
            }, f)
        print(f"Checkpoint saved at generation {i+1}")
    
    # Check if termination criteria met
    if pipeline.is_terminated(state, best):
        break
  
  # Show result
  pipeline.show(state, best)
  
  # Save the final model
  model_path = os.path.join(checkpoint_dir, "final_model.pkl")
  with open(model_path, 'wb') as f:
      pickle.dump(best, f)
  print(f"Final model saved to {model_path}")
  
  return best

def load_checkpoint(checkpoint_path):
    """Load a checkpoint to resume training"""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint['state'], checkpoint['best'], checkpoint['generation']

def resume_training(checkpoint_path, checkpoint_dir="checkpoints", save_interval=2):
    """Resume training from a checkpoint"""
    state, best, start_gen = load_checkpoint(checkpoint_path)
    
    # Recreate the pipeline (must match the original configuration)
    custom_problem = CustomFuncFit(
        func=pagie_polynomial,
        low_bounds=[-1, -1],
        upper_bounds=[1, 1],
        method="sample",
        num_samples=100,
    )
    
    def square(x):
        return x ** 2
    ACT.add_func("square", square)
    
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
        algorithm=algorithm,
        problem=custom_problem,
        generation_limit=10,
        fitness_target=-1e-4,
        seed=42,
    )
    
    # Continue training
    for i in range(start_gen, pipeline.generation_limit):
        state, step_best = pipeline.step(state)
        
        # Update best if needed
        if step_best.fitness > best.fitness:
            best = step_best
        
        # Save checkpoint
        if (i + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_gen_{i+1}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'state': state,
                    'best': best,
                    'generation': i+1
                }, f)
            print(f"Checkpoint saved at generation {i+1}")
        
        # Check if termination criteria met
        if pipeline.is_terminated(state, best):
            break
    
    # Show result
    pipeline.show(state, best)
    
    # Save the final model
    model_path = os.path.join(checkpoint_dir, "final_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best, f)
    print(f"Final model saved to {model_path}")
    
    return best

def use_trained_model(model_path, inputs):
    """Load and use a trained model to make predictions"""
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make a prediction
    prediction = model.predict(jnp.array(inputs))
    return prediction

if __name__ == "__main__":
    # Train a new model
    # trained_model = train()
    
    # Or resume training from a checkpoint
    # trained_model = resume_training("checkpoints/checkpoint_gen_6.pkl")
    
    # Or use a previously trained model
    # model_path = "checkpoints/final_model.pkl"
    # test_input = [0.5, 0.5]
    # prediction = use_trained_model(model_path, test_input)
    # print(f"Model prediction for input {test_input}: {prediction}")
    
    # Uncomment the desired operation
    train()