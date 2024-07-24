# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import pandas as pd
import numpy as np

q_table = pd.DataFrame(np.zeros((3, 3)), columns=['R', 'P', 'S'], index=['R', 'P', 'S'])

def player(prev_play, opponent_history=[], q_table=q_table, actions=[], n=[0]):
    opponent_history.append(prev_play)
    n[0] += 1

    if not opponent_history[-1] or not opponent_history[-2]:
        actions.append('R')
        return 'R'

    learning_rate = 0.7
    gamma = 0.3
    max_e = 1.0
    min_e = 0.01
    decay_rate = 0.005
    epsilon = min_e + (max_e - min_e) * np.exp(-decay_rate * n[0])
    
    state = opponent_history[-2]
    next_state = opponent_history[-1]
    action = actions[-1]
    if action == next_state:
        reward = 0
    elif action == 'R' and next_state == 'S' or action == 'P' and next_state == 'R' or action == 'S' and next_state == 'P':
        reward = 1
    else:
        reward = -1

    q_table.loc[state, action] = q_table.loc[state, action] + learning_rate * (reward + gamma * q_table.loc[next_state].max() - q_table.loc[state, action])

    random_n = np.random.rand()
    if random_n > epsilon:
        next_action = q_table.columns[q_table[next_state].argmax()]
    else:
        next_action = q_table.columns[np.random.randint(3)]

    actions.append(next_action)
    return next_action
