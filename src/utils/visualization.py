"""Visualization utilities for rendering simulations and creating videos."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mediapy as media
import logging
from typing import List, Optional
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from moojoco.mjcf.component import MJCFRootComponent


def render_mjcf_model(mjcf: MJCFRootComponent, mujoco) -> None:
    """Visualize a MuJoCo model from MJCF specification.
    
    Args:
        mjcf: The MJCF root component to render
        mujoco: The MuJoCo module
    """
    model = mujoco.MjModel.from_xml_string(mjcf.get_mjcf_str())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    for _ in range(100):  
        mujoco.mj_step(model, data)
        renderer.update_scene(data)

    img = renderer.render()
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def filter_valid_frames(images: List[np.ndarray]) -> List[np.ndarray]:
    """Filter out None frames from a list of images.
    
    Args:
        images: List of image frames, possibly containing None values
        
    Returns:
        List of valid image frames
    """
    valid_images = [image for image in images if image is not None]
    num_nones = len(images) - len(valid_images)
    
    if num_nones > 0:
        logging.warning(
            f"Rendering produced {num_nones} None frames. Video may be choppy."
        )
        
    return valid_images


def display_video_mediapy(images: List[np.ndarray]) -> Optional[str]:
    """Display video using mediapy.
    
    Args:
        images: List of frames to display
        
    Returns:
        String representation of the video or None
    """
    valid_frames = filter_valid_frames(images)
    if not valid_frames:
        logging.warning("No valid frames to display.")
        return None
        
    return media.show_video(images=valid_frames)


def display_video_matplotlib(images: List[np.ndarray]) -> None:
    """Display video using matplotlib animation.
    
    Args:
        images: List of frames to display
    """
    valid_frames = filter_valid_frames(images)
    
    if not valid_frames:
        print("No frames to display.")
        return

    fig, ax = plt.subplots()
    ax.axis("off")
    img_display = ax.imshow(valid_frames[0])

    def update(frame):
        img_display.set_array(frame)
        return (img_display,)

    ani = animation.FuncAnimation(
        fig, update, frames=valid_frames, interval=50, blit=True
    )
    plt.show()


def process_multi_camera_frames(
    render_output: List[np.ndarray],
    environment_configuration: MuJoCoEnvironmentConfiguration,
) -> np.ndarray:
    """Process multi-camera render output to create a combined frame.
    
    Args:
        render_output: List of frames from multiple cameras
        environment_configuration: Environment configuration with camera settings
        
    Returns:
        Combined frame with all camera views
    """
    if render_output is None:
        return None

    num_cameras = len(environment_configuration.camera_ids)
    num_envs = len(render_output) // num_cameras

    if num_cameras > 1:
        # Horizontally stack frames of the same environment
        frames_per_env = np.array_split(render_output, num_envs)
        render_output = [
            np.concatenate(env_frames, axis=1) for env_frames in frames_per_env
        ]

    # Vertically stack frames of different environments
    render_output = np.concatenate(render_output, axis=0)

    return render_output[:, :, ::-1]  # RGB to BGR


def save_initial_frame(initial_frame, output_path="initial_state.png", show=True):
    """Save and optionally display the initial frame of a simulation.
    
    Args:
        initial_frame: The initial frame to save
        output_path: Path where to save the frame
        show: Whether to display the frame
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(initial_frame)
    plt.axis("off")

    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def save_frame_sequence(frames, output_dir="simulation_output", sample_rate=10):
    """Save selected frames from a simulation as individual images.
    
    Args:
        frames: List of simulation frames
        output_dir: Directory to save the frames
        sample_rate: Save every nth frame
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(frames[::sample_rate]):
        plt.figure(figsize=(10, 8))
        plt.imshow(frame)
        plt.axis("off")
        plt.savefig(f"{output_dir}/frame_{i:03d}.png")
        plt.close()


def create_mp4_video(
    frames, output_path="simulation_output/simulation.mp4", fps=30
):
    """Create an MP4 video from a sequence of frames.
    
    Args:
        frames: List of frames
        output_path: Path where to save the video
        fps: Frames per second for the video
    """
    valid_frames = filter_valid_frames(frames)
    
    if not valid_frames:
        logging.warning("No frames to save.")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    media.write_video(path=output_path, images=valid_frames, fps=fps)
    print(f"Video saved to: {output_path}")


def save_and_display_video(
    frames, output_path=None, fps=30
):
    """Save and display a video from frames.
    
    Args:
        frames: List of frames
        output_path: Optional path to save the video
        fps: Frames per second
        
    Returns:
        Display object from mediapy
    """
    valid_frames = filter_valid_frames(frames)
    
    if output_path:
        create_mp4_video(valid_frames, output_path, fps)
        
    return media.show_video(images=valid_frames, fps=fps)
