"""Rendering utilities for the brittle star simulation."""
import os 

from utils.visualization import (
    display_video_mediapy as show_video,
    process_multi_camera_frames as post_render,
    save_initial_frame as visualize_initial_frame,
    save_frame_sequence as save_frame_samples,
    create_mp4_video,
    save_and_display_video
)


def play_video(video_path):
    """Play a video using the system's default video player"""
    import subprocess
    import platform

    abs_path = os.path.abspath(video_path)

    if platform.system() == "Darwin":
        subprocess.call(["open", abs_path])
    elif platform.system() == "Linux":
        subprocess.call(["xdg-open", abs_path])
    elif platform.system() == "Windows":
        os.startfile(abs_path)

# Re-export visualization functions for backward compatibility
__all__ = [
    "show_video",
    "post_render",
    "visualize_initial_frame",
    "save_frame_samples",
    "create_mp4_video",
    "save_and_display_video",
]