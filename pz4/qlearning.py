import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import time

GAMMA = 0.8
EPISODES = 1000
TARGET_STATE = 8

adj_matrix = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
])

R = -1 * np.ones_like(adj_matrix)
R[adj_matrix == 1] = 0
R[:, TARGET_STATE][adj_matrix[:, TARGET_STATE] == 1] = 100

Q = np.zeros_like(R)

print("Матриця R (винагород):\n", R.astype(int))

def get_available_actions(state):
    return np.where(adj_matrix[state] == 1)[0]

def choose_action(state):
    actions = get_available_actions(state)
    if random.random() < 0.2:
        return random.choice(actions)
    q_vals = Q[state, actions]
    max_q = np.max(q_vals)
    return random.choice(actions[q_vals == max_q])

for episode in range(EPISODES):
    state = random.randint(0, 8)
    while state != TARGET_STATE:
        action = choose_action(state)
        next_state = action
        max_future_q = np.max(Q[next_state])
        Q[state, action] = R[state, action] + GAMMA * max_future_q
        state = next_state

print("\nМатриця Q після навчання (сирі значення):\n", np.round(Q, 1))

Q_norm = Q / np.max(Q) * 100

def visualize_graph(Q, path=None):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)
    labels = {i: str(i) for i in G.nodes}

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=700, arrows=True)

    edge_labels = {(i, j): int(Q[i, j]) for i in range(9) for j in range(9) if Q[i, j] > 0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=3)
    plt.title("Вивчений граф переходів")
    plt.show()

visualize_graph(Q_norm)

def get_optimal_path(start_state):
    path = [start_state]
    current = start_state
    while current != TARGET_STATE:
        next_actions = get_available_actions(current)
        q_vals = Q[current, next_actions]
        best_action = next_actions[np.argmax(q_vals)]
        path.append(best_action)
        current = best_action
        if len(path) > 20:
            break
    return path

start_state = 0
optimal_path = get_optimal_path(start_state)
print("\nОптимальний шлях з", start_state, "до", TARGET_STATE, ":", [int(x) for x in optimal_path])
visualize_graph(Q_norm, path=optimal_path)

def animate_path(path):
    for i, state in enumerate(path):
        print(f"Крок {i}: агент у стані {state}")
        time.sleep(0.5)

print("\nРух агента:")
animate_path(optimal_path)
