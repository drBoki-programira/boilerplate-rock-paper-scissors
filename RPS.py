# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import pandas as pd
import numpy as np

MOVES = ('R', 'P', 'S')
index = [a + b for a in MOVES for b in MOVES]

q_table = pd.DataFrame(np.zeros((len(index), 3)), columns=MOVES, index=index)

def player(prev_play, opponent_history=[], q_table=q_table, actions=[]):
    opponent_history.append(prev_play)

    if not all(opponent_history[-2:]):
        first_play = 'S'
        actions.append(first_play)
        return first_play

    learning_rate = 0.7
    gamma = 0.3
    decay_rate = 0.99
    epsilon = 0.83
    
    state = actions[-2] + opponent_history[-2]
    next_state = actions[-1] + opponent_history[-1]
    action = actions[-1]
    
    if action == next_state[-1]:
        reward = 0
    elif action == 'R' and next_state[-1] == 'S' or action == 'P' and next_state[-1] == 'R' or action == 'S' and next_state[-1] == 'P':
        reward = 1
    else:
        reward = -1

    q_table.loc[state, action] = q_table.loc[state, action] + learning_rate * (reward + gamma * q_table.loc[next_state].max() - q_table.loc[state, action])

    random_n = np.random.rand()
    if epsilon > random_n:
        next_action = q_table.columns[q_table.loc[next_state].argmax()]
    else:
        next_action = q_table.columns[np.random.randint(3)]

    epsilon *= decay_rate
    actions.append(next_action)
    return next_action
