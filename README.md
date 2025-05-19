# Brittle Star Locomotion with NEAT

This project focuses on simulating and evolving locomotion strategies for a brittle star using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The simulation is built upon the MuJoCo physics engine and the `biorobot` library.

## Project Overview

The primary goal is to train a neural network controller that enables a simulated brittle star to perform directed locomotion. The NEAT algorithm is employed to evolve both the topology and weights of these neural networks. The project includes functionalities for:

*   Defining the brittle star's morphology (physical structure).
*   Creating and managing the simulation environment.
*   Implementing the NEAT algorithm for training controllers using [TensorNEAT](https://github.com/EMI-Group/tensorneat).
*   Visualizing the simulation and the performance of trained models.
*   Running experiments on HPC clusters.

## File Structure

Here's a breakdown of the key files and directories:

```
.
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── src/
    ├── NEAT/                
    │   ├── train.py              # Main script to start NEAT training
    │   ├── config.py             # Configuration for simulation and NEAT parameters
    │   ├── neat_problem.py       # Defines the NEAT problem for the brittle star
    │   ├── neat_controller.py    # NEAT controller logic
    │   └── visualize.py          # Utilities for visualizing NEAT results
    ├── environment.py            # Defines the simulation environment for the brittle star
    ├── morphology.py             # Defines the physical structure (morphology)
    ├── initialize.py             # Handles GPU and MuJoCo initialization
    ├── render.py                 # Functions for rendering simulations and creating videos
    └── hpc_script.sh             # Script for running training on an HPC cluster 
```

## Getting Started

### Prerequisites

*   Python 3.x
*   MuJoCo physics engine installed and configured.
*   Access to a GPU is recommended for faster simulations (see `src/initialize.py` for EGL setup).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Bendemeurichy/SEL3_2025_Groep_4.git
    cd SEL3_2025_Groep_4
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
    *   Modify parameters for the simulation, brittle star morphology, and NEAT algorithm in `config.py`. This includes settings like the number of arms, segments per arm, simulation duration, population size, etc.

2.  **Training a NEAT Controller:**
    *   To start the training process, run the `train.py` script within the `NEAT/` directory:
        ```bash
        python src/NEAT/train.py --mode <curriculum|no_curriculum> 
        ```
    *   Trained models (genomes) will be saved in the `./models/` directory.

3.  **Running on an HPC Cluster (PBS):**
    *   The `src/hpc_script.sh` is provided for running training jobs on an HPC cluster that uses the PBS scheduler.
    *   You might need to modify the script to match your cluster's environment (e.g., module loading, conda environment name).
    *   Submit the job using `qsub src/hpc_script.sh`.

4.  **Visualization and Testing:**
    *   `src/NEAT/visualize.py` can be used to load and visualize the behavior of trained genomes.
    *   Rendered videos are typically saved in the `./output_videos/` directory.


