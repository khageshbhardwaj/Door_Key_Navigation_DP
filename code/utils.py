import os
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import random
from minigrid.core.world_object import Goal, Key, Door, Wall

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def step_cost(action):
    # You should implement the stage cost by yourself
    # Feel free to use it or not
    # ************************************************
    return 1  # the cost of action


def step(env, action):
    """
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    """
    actions = {
        0: env.unwrapped.actions.forward,
        1: env.unwrapped.actions.left,
        2: env.unwrapped.actions.right,
        3: env.unwrapped.actions.pickup,
        4: env.unwrapped.actions.toggle,
    }

    (obs, reward, terminated, truncated, info) = env.step(actions[action])

    return step_cost(action), terminated

def get_current_state(env, info):
    """
    Retrieve the current state after performing the action in the environment.

    Parameters:
        env (gym.Env): The Gym environment.
        info (dict): Additional information about the environment.

    Returns:
        x (int): The x-coordinate of the agent's position.
        y (int): The y-coordinate of the agent's position.
        heading (int): The heading direction of the agent.
        door_state (bool): The state of the door (open or closed).
        key_state (bool): Whether the agent is carrying a key or not.
    """
    (x, y) = env.unwrapped.agent_pos
    heading = env.unwrapped.agent_dir

    door_state = env.unwrapped.grid.get(info["door_pos"][0], info["door_pos"][1]).is_open
    key_state = env.unwrapped.carrying is not None

    return x, y, heading, door_state, key_state


def generate_random_env(seed, task):
    """
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    """
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def load_env(path):
    """
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    with open(path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height, 
        "width": env.unwrapped.width, 
        "init_agent_pos": env.unwrapped.agent_pos, 
        "init_agent_dir": env.unwrapped.dir_vec,
        "wall_pos": [],
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.unwrapped.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Door):
                info["door_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Wall):
                info["wall_pos"].append(tuple([j, i]))

    return env, info


def load_random_env(env_folder):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder)]
    env_path = random.choice(env_list)
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height,
        "width": env.unwrapped.width,
        "init_agent_pos": env.unwrapped.agent_pos,
        "init_agent_dir": env.unwrapped.dir_vec,
        "door_pos": [],
        "door_open": [],
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Door):
                info["door_pos"].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info["door_open"].append(True)
                else:
                    info["door_open"].append(False)
            elif isinstance(env.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info, env_path


def save_env(env, path):
    with open(path, "wb") as f:
        pickle.dump(env, f)


def plot_env(env):
    """
    Plot current environment
    ----------------------------------
    """
    img = env.render()
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_gif_from_seq(seq, env, path="./gif/doorkey.gif", duration=0.8, loop=True):
    """
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]

    env:
        The doorkey environment
    """
    with imageio.get_writer(path, mode="I", duration=0.8) as writer:
        img = env.render()
        writer.append_data(img)
        for act in seq:
            img = env.render()
            step(env, act)
            writer.append_data(img)
    print(f"GIF is written to {path}")
    return
    

def draw_gif_from_seq1(seq, env, path="./gif/doorkey.gif", last_image_path="./gif/last_image.png", duration=0.8, loop=True):
    """
    Save gif with a given action sequence and save the last image of the sequence.
    ----------------------------------------------------------------
    seq:
        Action sequence, e.g., [0,0,0,0] or ['MF', 'MF', 'MF', 'MF']

    env:
        The doorkey environment

    path:
        Path where the gif should be saved

    last_image_path:
        Path where the last image should be saved
    """
    with imageio.get_writer(path, mode='I', duration=duration) as writer:
        img = env.render()
        writer.append_data(img)
        for act in seq:
            img = env.render()
            step(env, act)  # This function should perform the action on the environment
            writer.append_data(img)
        
        # Save the last image separately
        imageio.imwrite(last_image_path, img)
        print(f"Last image is written to {last_image_path}")

    print(f"GIF is written to {path}")
    return
