import mujoco_py
import numpy as np

def create_flat_environment():
    model = mujoco_py.load_model_from_path("flat_environment.xml")
    sim = mujoco_py.MjSim(model)
    return sim

def choose_destination():
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    return x, y

def spawn_brittle_star(sim):
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    sim.data.qpos[0] = x
    sim.data.qpos[1] = y

if __name__ == "__main__":
    sim = create_flat_environment()
    destination = choose_destination()
    spawn_brittle_star(sim)
    viewer = mujoco_py.MjViewer(sim)
    while True:
        sim.step()
        viewer.render()
