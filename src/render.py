from initialize import mujoco
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from moojoco.mjcf.component import MJCFRootComponent
import logging

from typing import List
import numpy as np
import mediapy as media
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
import os


def visualize_mjcf(mjcf: MJCFRootComponent) -> None:
    model = mujoco.MjModel.from_xml_string(mjcf.get_mjcf_str())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    for _ in range(100):  # Simulate for 100 steps
        mujoco.mj_step(model, data)
        renderer.update_scene(data)

    img = renderer.render()

    # Display using matplotlib
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()


def show_video(images: List[np.ndarray | None]) -> str | None:
    # Temporary workaround until https://github.com/google-deepmind/mujoco/issues/1379 is fixed
    filtered_images = [image for image in images if image is not None]
    num_nones = len(images) - len(filtered_images)
    if num_nones > 0:
        logging.warning(
            f"env.render produced {num_nones} None's. Resulting video might be a bit choppy (consquence of https://github.com/google-deepmind/mujoco/issues/1379)."
        )
    return media.show_video(images=filtered_images)


def show_video_2(images: List[np.ndarray | None]) -> None:
    filtered_images = [image for image in images if image is not None]

    if not filtered_images:
        print("No frames to display.")
        return

    fig, ax = plt.subplots()
    ax.axis("off")
    img_display = ax.imshow(filtered_images[0])

    def update(frame):
        img_display.set_array(frame)
        return (img_display,)

    ani = animation.FuncAnimation(
        fig, update, frames=filtered_images, interval=50, blit=True
    )
    plt.show()


def post_render(
    render_output: List[np.ndarray],
    environment_configuration: MuJoCoEnvironmentConfiguration,
) -> np.ndarray:
    if render_output is None:
        return

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


def visualize_initial_frame(initial_frame, output_path="initial_state.png", show=True):
    """Display and optionally save the initial frame"""
    plt.figure(figsize=(10, 8))
    plt.imshow(initial_frame)
    plt.axis("off")

    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def save_frame_samples(frames, output_dir="simulation_output", sample_rate=10):
    """Save selected frames from the simulation as images"""
    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(frames[::sample_rate]):
        plt.figure(figsize=(10, 8))
        plt.imshow(frame)
        plt.axis("off")
        plt.savefig(f"{output_dir}/frame_{i:03d}.png")
        plt.close()


def create_animation(
    frames, output_path="simulation_output/brittle_star_simulation.mp4", interval=33, high_quality=False
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    plt.axis("off")

    def init():
        im = plt.imshow(frames[0])
        plt.tight_layout(pad=0)
        return [im]

    def animate(i):
        plt.clf()
        plt.axis("off")
        im = plt.imshow(frames[i])
        plt.tight_layout(pad=0)
        return [im]

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(frames), interval=interval, blit=True
    )

    dpi = 200 if high_quality else 100
    bitrate = 5000 if high_quality else 1800
    fps = 30 if high_quality else 20
    
    ani.save(
        output_path, 
        fps=fps, 
        dpi=dpi,
        bitrate=bitrate
    )
    return ani


def play_video(video_path):
    """Play a video using the system's default video player"""
    import subprocess
    import platform

    abs_path = os.path.abspath(video_path)

    if platform.system() == "Darwin":  # macOS
        subprocess.call(["open", abs_path])
    elif platform.system() == "Linux":
        subprocess.call(["xdg-open", abs_path])
    elif platform.system() == "Windows":
        os.startfile(abs_path)
