"""Rendering utilities for the brittle star simulation."""
from initialize_clean import mujoco
from utils.visualization import (
    render_mjcf_model,
    display_video_mediapy as show_video,
    display_video_matplotlib as show_video_matplotlib,
    process_multi_camera_frames as post_render,
    save_initial_frame as visualize_initial_frame,
    save_frame_sequence as save_frame_samples,
    create_mp4_video,
    save_and_display_video
)

# Re-export visualization functions for backward compatibility
__all__ = [
    "visualize_mjcf",
    "show_video",
    "show_video_2",
    "post_render",
    "visualize_initial_frame",
    "save_frame_samples",
    "create_animation",
    "save_and_display_video",
]

# For backward compatibility
visualize_mjcf = lambda mjcf: render_mjcf_model(mjcf, mujoco)
show_video_2 = show_video_matplotlib
create_animation = create_mp4_video
