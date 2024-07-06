from utils import *
import gymnasium as gym
from minigrid.core.world_object import Goal, Key, Door, Wall
# %load_ext autoreload
# %autoreload 2

def motion_model(current_state, action, env_info, info):

    # Extract necessary information from the environment info dictionary
    height = info['height']
    width = info['width']
    key_pos = info['key_pos']
    door_pos = info['door_pos']
    goal_pos = info['goal_pos']


    # Unpack current state
    x, y, heading, key_avail, door_state = current_state
    updated_state = current_state

    # Define movement vectors for each heading direction
    movement_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Right, Down, Left, Up

    # next state
    dx, dy = movement_vectors[heading]
    new_x, new_y = x + dx, y + dy

    if action == 0:  # Move forward
        # Check if the new position is within the grid boundaries
        if 0 <= new_x < width and 0 <= new_y < height:
            cell_obj = env_info.grid.get(new_x, new_y)
            if cell_obj is not None:
                if cell_obj.type == "wall":
                    return current_state

                elif new_x == door_pos[0] and new_y == door_pos[1] and door_state == 0:
                    return current_state

                else:
                    updated_state[0] = new_x
                    updated_state[1] = new_y
                    return updated_state

            else:
                updated_state[0] = new_x
                updated_state[1] = new_y
                return updated_state

        else:
            return current_state

    elif action == 1:  # Turn Left
        updated_state[2] = (heading - 1) % 4
        return updated_state

    elif action == 2:  # Turn Right
        updated_state[2] = (heading + 1) % 4
        return updated_state

    elif action == 3:
        if key_avail == 0 and new_x == key_pos[0] and new_y == key_pos[1]:
            updated_state[3] = 1
            return updated_state
        else:
            return current_state

    elif action == 4:
        if key_avail == 1 and new_x == door_pos[0] and new_y == door_pos[1]:
            updated_state[4] = (door_state +1 ) % 2
            return updated_state

        else:
            return current_state

    else:
        return current_state

def stage_cost(x, y, goal_pos):

    if x == goal_pos[0] and y == goal_pos[1]:
        cost = 0
    else:
        cost = 1

    return cost

# partA

"""
    You are required to find the optimal path in
    doorkey-5x5-normal.env
    doorkey-6x6-normal.env
    doorkey-8x8-normal.env

    doorkey-6x6-direct.env
    doorkey-8x8-direct.env

    doorkey-6x6-shortcut.env
    doorkey-8x8-shortcut.env
"""
env_dict = {
    1: './envs/known_envs/doorkey-5x5-normal.env',
    2: './envs/known_envs/doorkey-6x6-normal.env',
    3: './envs/known_envs/doorkey-8x8-normal.env',
    4: './envs/known_envs/doorkey-6x6-direct.env',
    5: './envs/known_envs/doorkey-8x8-direct.env',
    6: './envs/known_envs/doorkey-6x6-shortcut.env',
    7: './envs/known_envs/doorkey-8x8-shortcut.env'
}

env, info = load_env(env_dict[3])  # load an environment

# possible actions
MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

actions = [MF, TL, TR, PK, UD]

# environment parameters
height = info['height']
width = info['width']
heading = [0, 1, 2, 3]  #0: Right, 1: Down, 2: Left, 3: Up
key_state = [0, 1]  # 0: Not having the key, 1: Having the key
door_state = [0, 1]  # 0: Closed, 1: Open

key_location = info['key_pos']
goal_location = info['goal_pos']


Value_function = np.full((width, height, len(heading), len(key_state), len(door_state)), np.inf)  # Value function
Value_function_T = np.full((width, height, len(heading), len(key_state), len(door_state)), np.inf)
Value_function[goal_location[0], goal_location[1], :, :, :] = 0

policy = np.full((width, height, len(heading), len(key_state), len(door_state)), None)  # Policy


# time horizon
T = 50
Q = np.zeros(len(actions))
gamma = 1

for t in range(T-1, -1, -1):
    for x in range(width):
        for y in range(height):
            for direction in heading:  # [0, 1, 2, 3]
                for is_key_avail in key_state:    # (0: not having the key, 1: having the key)
                    for is_door_open in door_state:  # (0: closed, 1: open)
                        for act in actions:   #[0, 1, 2, 3, 4]
                            state = [x, y, direction, is_key_avail, is_door_open]
                            next_state = motion_model(state, act, env, info)
                            stg_cost = stage_cost(x, y, goal_location)
                            next_x, next_y, next_heading, next_isKey, next_isDoorOpen = next_state
                            Q[act] = stg_cost + gamma * Value_function[next_x, next_y, next_heading, next_isKey, next_isDoorOpen]

                        Value_function_T[x, y, direction, is_key_avail, is_door_open] = np.min(Q)
                        policy[x, y, direction, is_key_avail, is_door_open] = np.argmin(Q)
    Value_function = Value_function_T

########################################################################################################

# Optmial Policy

########################################################################################################

# Initial Parameters
x = info['init_agent_pos'][0]
y = info['init_agent_pos'][1]
heading = env.unwrapped.agent_dir
is_key = 0
door_st = 0

# Define a dictionary for action names
action_names = {0: 'MF', 1: 'TL', 2: 'TR', 3: 'PK', 4: 'UD'}

state = [x, y, heading, is_key, door_st]
optimum_policy = []
optimum_policy_digit = []

print('Agent\'s State: [x, y, heading, is_key_available, is_door_open]')

while True:
    next_x, next_y, next_heading, next_isKey, next_isDoorOpen = state
    act = policy[next_x, next_y, next_heading, next_isKey, next_isDoorOpen]
    action_name = action_names[act]  # Convert action number to name using the dictionary
    state = motion_model(state, act, env, info)  # Update the state
    print(state)
    optimum_policy.append(action_name)  # Append the action name instead of number
    optimum_policy_digit.append(act)  # Append action number to create GIF

    if state[0] == info['goal_pos'][0] and state[1] == info['goal_pos'][1]:
        # optimum_policy.append(action_name)
        optimum_policy_digit.append(act)

        break

print('Optimum Policy: ', optimum_policy)

# create gif
draw_gif_from_seq1(optimum_policy_digit, env)  # draw a GIF & save

