# Brittle Star Locomotion with NEAT

This project focuses on simulating and evolving locomotion strategies for a virtual brittle star using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The simulation is built upon the MuJoCo physics engine and the `biorobot` library.

## Project Overview

The primary goal is to train a neural network controller that enables a simulated brittle star to perform tasks like directed or undirected locomotion. The NEAT algorithm is employed to evolve both the topology and weights of these neural networks. The project includes functionalities for:

*   Defining the brittle star's morphology (physical structure).
*   Creating and managing the simulation environment.
*   Implementing the NEAT algorithm for training controllers.
*   Visualizing the simulation and the performance of trained models.
*   Running experiments on High-Performance Computing (HPC) clusters.

## File Structure

Here's a breakdown of the key files and directories:

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── src/
    ├── environment.py        # Defines the simulation environment for the brittle star
    ├── morphology.py         # Defines the physical structure (morphology)
    ├── initialize.py         # Handles GPU and MuJoCo initialization
    ├── render.py             # Functions for rendering simulations and creating videos
    ├── hpc_script.sh         # Script for running training on an HPC cluster (PBS)
    ├── NEAT_basic_test.py    # Basic test script for the NEAT setup and environment
    ├── NEAT_test.ipynb       # Jupyter Notebook for experimentation
    ├── NEAT/                   # Core NEAT algorithm implementation
    │   ├── train.py            # Main script to start NEAT training
    │   ├── config.py           # Configuration for simulation and NEAT parameters
    │   ├── neat_problem.py     # Defines the NEAT problem for the brittle star
    │   ├── neat_controller.py  # NEAT controller logic
    │   ├── visualize.py        # Utilities for visualizing NEAT results
    │   ├── NBestPipeline.py    # Custom implementation of tensorNEAT pipeline class that tracks n best genomes
    │   └── ...                 # Other NEAT-related modules
    ├── models/                 # Stores trained NEAT genomes (models)
    │   ├── best_genome.pkl     # Example of a saved best-performing genome
    │   └── curr_test_*/        # Directories for results from different training runs
    └── output_videos/          # Directory for rendered videos of simulations
```

## Getting Started

### Prerequisites

*   Python 3.x
*   MuJoCo physics engine installed and configured.
*   Access to a GPU is recommended for faster simulations (see `src/initialize.py` for EGL setup).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    It's highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

### Running the Simulation and Training

1.  **Configuration:**
    *   Modify parameters for the simulation, brittle star morphology, and NEAT algorithm in `src/NEAT/config.py`. This includes settings like the number of arms, segments per arm, simulation duration, population size, etc.

2.  **Training a NEAT Controller:**
    *   To start the training process, run the `train.py` script within the `src/NEAT/` directory:
        ```bash
        python src/NEAT/train.py
        ```
    *   The script supports different training modes, including curriculum learning (e.g., `train_neat_curriculum()`). Check the `if __name__ == "__main__":` block in `src/NEAT/train.py` to see which training function is currently active.
    *   Trained models (genomes) will be saved in the `src/models/` directory.
    *   The script has arguments for the training mode with `--mode {"curriculum" / "no_curriculum"}` and for an identifier for the produced model files with `--index {integer}` 

3.  **Running on an HPC Cluster (PBS):**
    *   The `src/hpc_script.sh` is provided for running training jobs on an HPC cluster that uses the PBS scheduler.
    *   You might need to modify the script to match your cluster's environment (e.g., module loading, conda environment name).
    *   Submit the job using `qsub src/hpc_script.sh`.

4.  **Visualization and Testing:**
    *   `src/render.py` contains functions to visualize MJCF models and simulation frames.
    *   `src/NEAT/visualize.py` can be used to load and visualize the behavior of trained genomes.
    *   `src/NEAT_basic_test.py` provides a script to test the basic environment setup and rendering.
    *   Rendered videos are typically saved in the `src/output_videos/` directory.

## Key Scripts

*   **`src/NEAT/train.py`**: The main entry point for training NEAT controllers.
*   **`src/NEAT/config.py`**: Central configuration file for most parameters.
*   **`src/environment.py`**: Defines the interaction logic between the agent and the simulated world.
*   **`src/morphology.py`**: Defines how the brittle star is constructed.
*   **`src/render.py` & `src/NEAT/visualize.py`**: For visualizing simulations and results.

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
