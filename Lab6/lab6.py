import math
import matplotlib.pyplot as plt
import numpy as np

MAXSIZE = float("inf")

def generate_cities(N: int, radius: float = 10, noise_level: float = 42) -> list[tuple[float, float]]:
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles += np.random.uniform(-noise_level, noise_level, size=N)
    
    return [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]

def distance_matrix(cities: list[tuple[float, float]]):
    N = len(cities)
    dist_mat: np.ndarray = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            dist_mat[i][j] = math.sqrt((cities[i][0] - cities[j][0])**2 + (cities[i][1] - cities[j][1])**2)

    return dist_mat

def copy_to_final(curr_path, final_path, N):
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]

def first_min(adj: list[tuple[float, float]], i: int) -> float:
    min = MAXSIZE
    for j in range(N):
        if adj[i][j] < min and i != j:
            min = adj[i][j]

    return min

def second_min(adj: list[tuple[float, float]], i: int) -> float:
    first = second = MAXSIZE

    for j in range(N):
        if i == j:
            continue

        if adj[i][j] <= first:
            second = first
            first = adj[i][j]
        elif adj[i][j] <= second and adj[i][j] != first:
            second = adj[i][j]

    return second

def tsp_recursive(adj, curr_bound, curr_weight, level, curr_path, visited, final_path, N):
    global final_res

    if level == N:
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
            curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]]
            if curr_res < final_res:
                copy_to_final(curr_path, final_path, N)
                final_res = curr_res
        return
    
    for i in range(N):
        if adj[curr_path[level-1]][i] != 0 and not visited[i]:
            curr_weight += adj[curr_path[level - 1]][i]

            if level == 1:
                curr_bound -= ((first_min(adj, curr_path[level - 1]) +
                                first_min(adj, i)) / 2)
            else:
                curr_bound -= ((second_min(adj, curr_path[level - 1]) +
                                first_min(adj, i)) / 2)
                
            if curr_bound + curr_weight < final_res:
                visited[i] = True
                curr_path[level] = i
                tsp_recursive(adj, curr_bound, curr_weight, level + 1, curr_path, visited, final_path, N)
                visited[i] = False

            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound += ((second_min(adj, curr_path[level - 1]) +
                            first_min(adj, i)) / 2)

def branch_and_bound_tsp(initial_path, dist_mat, N):
    global final_res
    final_res = MAXSIZE

    curr_bound = 0
    curr_path = [-1] * (N + 1)
    visited = [False] * N

    for i in range(N):
        curr_bound += (first_min(dist_mat, i) + second_min(dist_mat, i))
    curr_bound = math.ceil(curr_bound / 2)

    curr_path[0] = initial_path[0]
    visited[0] = True

    final_path = [-1] * (N + 1)
    tsp_recursive(dist_mat, curr_bound, 0, 1, curr_path, visited, final_path, N)

    print("Minimum cost:", final_res)
    print("Path taken:", final_path)
    
def visualize_cities(cities: list[tuple[float, float]]) -> None:
    x_values = [city[0] for city in cities]
    y_values = [city[1] for city in cities]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_values, y_values, color="blue", label="Miasta")

    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=12, ha="right")

    plt.title("Rozkład miast")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def visualize_route():
    # TODO
    pass

if __name__ == "__main__":
    N = 10
    d = 2

    cities = generate_cities(N, radius=10)

    visualize_cities(cities)

    dist_mat = distance_matrix(cities)

    initial_path: list[int] = [0]
    
    branch_and_bound_tsp(initial_path, dist_mat, N)