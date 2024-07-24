# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import pandas as pd
import numpy as np

index = []
for a in ['R', 'P', 'S']:
    for b in ['R', 'P', 'S']:
        for c in ['R', 'P', 'S']:
            index.append(a + b + c)

q_table = pd.DataFrame(np.zeros((len(index), 3)), columns=['R', 'P', 'S'], index=index)

def player(prev_play, opponent_history=[], q_table=q_table, actions=[], n=[0]):
    opponent_history.append(prev_play)
    n[0] += 1

    if not all(opponent_history[-4:]):
        first_few = q_table.columns[np.random.randint(3)]
        actions.append(first_few)
        for col in q_table.columns:
            q_table[col] = 0.0
        return first_few

    learning_rate = 0.4
    gamma = 0.97
    max_e = 1.0
    min_e = 0.02
    decay_rate = 0.008
    epsilon = min_e + (max_e - min_e) * np.exp(-decay_rate * n[0])
    
    state = ''.join(opponent_history[-4:-1])
    next_state = ''.join(opponent_history[-3:])
    action = actions[-1]
    if action == next_state[-1]:
        reward = 0
    elif action == 'R' and next_state[-1] == 'S' or action == 'P' and next_state[-1] == 'R' or action == 'S' and next_state[-1] == 'P':
        reward = 1
    else:
        reward = -1

    q_table.loc[state, action] = q_table.loc[state, action] + learning_rate * (reward + gamma * q_table.loc[next_state].max() - q_table.loc[state, action])

    random_n = np.random.rand()
    if random_n > epsilon:
        next_action = q_table.columns[q_table.loc[next_state].argmax()]
    else:
        next_action = q_table.columns[np.random.randint(3)]

    actions.append(next_action)
    return next_action
