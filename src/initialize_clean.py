"""System initialization module for MuJoCo and GPU setup."""
from utils.gpu_setup import initialize_system

# Initialize and export mujoco
mujoco = initialize_system()
