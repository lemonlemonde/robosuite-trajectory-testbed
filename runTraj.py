import os
import numpy as np
import robosuite as suite
from glob import glob
from robosuite.wrappers import DataCollectionWrapper
from robosuite.environments.manipulation.lift_modded import LiftModded




def load_trajectory():
    # Load the trajectory file
    trajs = np.load(path + "/train/trajs.npy")
    trajs_A = np.load(path + "/train/traj_a_indexes.npy")
    trajs_B = np.load(path + "/train/traj_b_indexes.npy")

    # read states back, load them one by one, and render
    t = 0
    for state in trajs[0]:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        env.render()
        t += 1
        if t % 100 == 0:
            print(t)


def collect_random_trajectory(env, timesteps=1000):
    """Run a random policy to collect trajectories.

    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment instance to collect trajectories from
        timesteps(int): how many environment timesteps to run for a given trajectory
    """

    env.reset()
    dof = env.action_dim

    for t in range(timesteps):
        action = np.random.randn(dof)
        env.step(action)
        env.render()
        if t % 100 == 0:
            print(t)


def playback_trajectory(env, ep_dir):
    """Playback data from an episode.

    Args:
        env (MujocoEnv): environment instance to playback trajectory in
        ep_dir (str): The path to the directory containing data for an episode.
    """

    # first reload the model from the xml
    xml_path = os.path.join(ep_dir, "model.xml")
    with open(xml_path, "r") as f:
        env.reset_from_xml_string(f.read())

    state_paths = os.path.join(ep_dir, "state_*.npz")

    # read states back, load them one by one, and render
    t = 0
    for state_file in sorted(glob(state_paths)):
        print(state_file)
        dic = np.load(state_file)
        states = dic["states"]
        print(states.shape)
        for state in states:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.render()
            t += 1
            if t % 100 == 0:
                print(t)


def test_collect_playback():
    # collect some data
    print("Collecting some random data...")
    collect_random_trajectory(env, timesteps=100)

    # playback some data
    _ = input("Press any key to begin the playback...")
    print("Playing back the data...")
    data_directory = env.ep_directory
    playback_trajectory(env, data_directory)











if __name__ == "__main__":

    path = "./56x3_expertx50_all-pairs_noise-augmentation10_id-mapping_with-videos_seed251/"

    # create environment instance
    env = suite.make(
        env_name="LiftModded",
        robots="Panda",
        has_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
    )
    data_directory = path + "./data/"

    # wrap the environment with data collection wrapper
    env = DataCollectionWrapper(env, data_directory)

    # reset the environment
    env.reset()

    test_collect_playback()