"""Environment creation and management for brittle star simulations."""
from utils.environment_factory import (
    create_arena,
    create_environment,
    create_default_environment_config as create_example_environment,
    create_default_arena_config
)
from morphology import create_morphology, default_brittle_star_morphology_specification

# Re-export functions for backward compatibility
__all__ = [
    "create_arena",
    "create_environment",
    "create_example_environment",
    "create_morphology",
    "default_brittle_star_morphology_specification"
]
