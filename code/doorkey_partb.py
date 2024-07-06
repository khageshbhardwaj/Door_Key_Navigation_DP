from utils import *
import gymnasium as gym
from minigrid.core.world_object import Goal, Key, Door, Wall
from tqdm import tqdm

# %load_ext autoreload
# %autoreload 2

def motion_model(current_state, key_pos, goal_pos, door_loc, action, env_info, info):

    # Unpack current state
    x, y, heading, key_avail, door1_state, door2_state = current_state
    updated_state = current_state

    door1_pos = door_loc[0]
    door2_pos = door_loc[1]


    # Define movement vectors for each heading direction
    movement_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Right, Down, Left, Up

    # next state
    dx, dy = movement_vectors[heading]
    new_x, new_y = x + dx, y + dy

    if action == 0:  # Move forward
        # Check if the new position is within the grid boundaries
        if 0 <= new_x < 8 and 0 <= new_y < 8:
            cell_obj = env_info.grid.get(new_x, new_y)
            if cell_obj is not None:
                if cell_obj.type == "wall":
                    return current_state

                elif new_x == door1_pos[0] and new_y == door1_pos[1] and door1_state == 0:
                    return current_state

                elif new_x == door2_pos[0] and new_y == door2_pos[1] and door2_state == 0:
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
        if key_avail == 1:  # Only try to unlock/lock doors if the key is available
            doors = [(door1_pos, 4), (door2_pos, 5)]  # List of door positions and their respective state indices
            for door_pos, state_idx in doors:
                if new_x == door_pos[0] and new_y == door_pos[1]:
                    updated_state[state_idx] = (updated_state[state_idx] + 1) % 2  # Toggle the door state
                    return updated_state
        return current_state  # Return current state if no door is unlocked

    else:
        return current_state

def stage_cost(x, y, goal_pos):
    if x == list(goal_pos)[0] and y == list(goal_pos)[1]:
        cost = 0
    else:
        cost = 1

    return cost

# partB

# load the random environment

env_folder = "./envs/random_envs"
env, info, env_path = load_random_env(env_folder)

# env, info = load_env('./envs/random_envs/DoorKey-8x8-12.env')  # load an environment

########################################################################################################

# Dynamic Programming

########################################################################################################

# possible actions
MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

# environment parameters
T = 50    # time horizon
width = 8
height = 8
heading = [0, 1, 2, 3]  #0: Right, 1: Down, 2: Left, 3: Up
key_state = [0, 1]  # 0: Not having the key, 1: Having the key
door1_state = [0, 1]  # 0: Closed, 1: Open
door2_state = [0, 1]  # 0: Closed, 1: Open
possible_key_location = [(1, 1), (2, 3), (1, 6)]
possible_goal_location = [(5, 1), (6, 3), (5, 6)]
actions = [MF, TL, TR, PK, UD]

door_loc = [(4, 2), (4, 5)]

key_location = info['key_pos']
goal_location = info['goal_pos']

Value_function = np.full((8, 8, 4, 2, 2, 2, 3, 3), np.inf)  # Value function
# Value_function[goal_location[0], goal_location[1], :, :, :, :, :, :] = 0     # value function at goal location
Value_function_T = np.full((8, 8, 4, 2, 2, 2, 3, 3), np.inf)
policy = np.full((8, 8, 4, 2, 2, 2, 3, 3), None)  # Policy


Value_function[info["goal_pos"][0],info["goal_pos"][1],:,:,:,:,:,:] = 0

Q = np.zeros(len(actions))
gamma = 1


for t in tqdm(range(T-1, -1, -1)):
    for x in range(width):
        for y in range(height):
            for direction in heading:
                for is_key_avail in key_state:
                    for is_door1_open in door1_state:
                        for is_door2_open in door2_state:
                            for key_pos_idx, key_pos in enumerate(possible_key_location):
                                for goal_pos_idx, goal_pos in enumerate(possible_goal_location):
                                    for act in actions:
                                        state = [x, y, direction, is_key_avail, is_door1_open, is_door2_open]
                                        next_state = motion_model(state, key_pos, goal_pos, door_loc, act, env, info)
                                        stg_cost = stage_cost(x, y, goal_pos)
                                        next_x, next_y, next_heading, next_isKey, next_isDoor1_Open, next_isDoor2_Open = next_state

                                        Q[act] = stg_cost + gamma * Value_function[next_x, next_y, next_heading, next_isKey, next_isDoor1_Open, next_isDoor2_Open, key_pos_idx, goal_pos_idx]

                                    Value_function_T[x, y, direction, is_key_avail, is_door1_open, is_door2_open, key_pos_idx, goal_pos_idx] = np.min(Q)
                                    policy[x, y, direction, is_key_avail, is_door1_open, is_door2_open, key_pos_idx, goal_pos_idx] = np.argmin(Q)
    Value_function = Value_function_T


########################################################################################################

# Optmial Policy

########################################################################################################

# Initial Parameters
x = 3
y = 5
heading = 3
is_key_avail = 0
is_door1_open = 1 if info['door_open'][0] else 0
is_door2_open = 1 if info['door_open'][1] else 0

# is_door1_open = 0
# is_door2_open = 0

door_loc = [(4, 2), (4, 5)]

# extracting the index of key location from possible key locations
possible_key_location = [(1, 1), (2, 3), (1, 6)]
key_pos = info["key_pos"]
key_pos_idx = possible_key_location.index(tuple(key_pos))

# extracting the index of goal location from possible goal locations
possible_goal_location = [(5, 1), (6, 3), (5, 6)]
goal_pos = info["goal_pos"]
goal_pos_idx = possible_goal_location.index(tuple(goal_pos))


# Define a dictionary for action names
action_names = {0: 'MF', 1: 'TL', 2: 'TR', 3: 'PK', 4: 'UD'}

state = [x, y, heading, is_key_avail, is_door1_open, is_door2_open]
optimum_policy = []
optimum_policy_digit = []

print('Agent\'s State: [x, y, heading, is_key_available, is_door1_open, is_door2_open]')

while True:
    next_x, next_y, next_heading, next_isKey, next_isDoor1_Open, next_isDoor2_Open = state
    act = policy[next_x, next_y, next_heading, next_isKey, next_isDoor1_Open, next_isDoor2_Open, key_pos_idx, goal_pos_idx]
    action_name = action_names[act]  # Convert action number to name using the dictionary
    state_1 = motion_model(state, key_pos, goal_pos, door_loc, act, env, info)
    print(state_1)
    optimum_policy.append(action_name)
    optimum_policy_digit.append(act)

    if state_1[0] == info['goal_pos'][0] and state_1[1] == info['goal_pos'][1]:
        # optimum_policy.append(action_name)
        optimum_policy_digit.append(act)

        break

print('Optimum Policy: ',optimum_policy)

# create gif
draw_gif_from_seq1(optimum_policy_digit, env)  # draw a GIF & save