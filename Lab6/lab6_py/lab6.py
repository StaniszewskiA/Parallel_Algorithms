import math
import matplotlib.pyplot as plt
import numpy as np
# from mpi4py import MPI
# from mpi4py.futures import MPIPoolExecutor

MAXSIZE = float("inf")

def generate_cities(N: int, radius: float = 10, noise_level: float = 42, generate_inside: bool = False) -> list[tuple[float, float]]:
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles += np.random.uniform(-noise_level, noise_level, size=N)
    
    if generate_inside:
        radii = np.random.uniform(0, radius, size=N)
    else:
        radii = np.full(N, radius)

    return [(radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i])) for i in range(N)]

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
    for j in range(len(adj)):
        if adj[i][j] < min and i != j:
            min = adj[i][j]
    return min

def second_min(adj: list[tuple[float, float]], i: int) -> float:
    first = second = MAXSIZE
    for j in range(len(adj)):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]
        elif adj[i][j] <= second and adj[i][j] != first:
            second = adj[i][j]
    return second

def tsp_recursive(adj, curr_bound, curr_weight, level, curr_path, visited, final_path, N, final_res):
    if level == N:
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
            curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]]
            if curr_res < final_res[0]:
                copy_to_final(curr_path, final_path, N)
                final_res[0] = curr_res
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
                
            if curr_bound + curr_weight < final_res[0]:
                visited[i] = True
                curr_path[level] = i
                tsp_recursive(adj, curr_bound, curr_weight, level + 1, curr_path, visited, final_path, N, final_res)
                visited[i] = False

            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound += ((second_min(adj, curr_path[level - 1]) +
                            first_min(adj, i)) / 2)

def branch_and_bound_tsp(initial_path, N):
    final_res = [MAXSIZE]
    curr_bound = 0
    curr_path = [-1] * (N + 1)
    visited = [False] * N

    dist_mat = distance_matrix(cities)

    for i in range(N):
        curr_bound += (first_min(dist_mat, i) + second_min(dist_mat, i))
    curr_bound = math.ceil(curr_bound / 2)

    curr_path[0] = 0
    visited[0] = True

    final_path = [-1] * (N + 1)
    tsp_recursive(dist_mat, curr_bound, 0, 1, curr_path, visited, final_path, N, final_res)

    return final_res[0], final_path

# def branch_and_bound_tsp_parallel(cities: list[tuple[float, float]], N: int, depth: int):
#     dist_mat = distance_matrix(cities)
#     initial_paths = [[0, i] for i in range(1, min(N, depth+1))]

#     def process_path(path: list[int]) -> tuple[float, list[int]]:
#         visited = [False] * N
#         visited[0] = True
#         curr_bound = 0
#         curr_path = [-1] * (N + 1)
#         curr_path[0] = path[0]
#         for i in range(N):
#             curr_bound += (first_min(dist_mat, i) + second_min(dist_mat, i))
#         curr_bound = math.ceil(curr_bound / 2)

#         final_res = [MAXSIZE]
#         final_path = [-1] * (N + 1)
#         tsp_recursive(dist_mat, curr_bound, 0, 1, curr_path, visited, final_path, N, final_res)
#         return final_res[0], final_path

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     if rank == 0:
#         initial_paths = comm.bcast(initial_paths, root=0)

#         results = []
#         for i, path in enumerate(initial_paths):
#             result = process_path(path)
#             results.append(result)

#         final_res = [MAXSIZE]
#         final_path = [-1] * (N + 1)
#         for res in results:
#             if res[0] < final_res[0]:
#                 final_res[0] = res[0]
#                 final_path = res[1]

#         return final_res[0], final_path

#     return None, None

def visualize_cities(cities: list[tuple[float, float]]) -> None:
    x_values = [city[0] for city in cities]
    y_values = [city[1] for city in cities]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_values, y_values, color="blue", label="Miasta")

    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=12, ha="right")

    plt.title("Rozkład miast")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def visualize_best_path(cities: list[tuple[float, float]], route: list[int]) -> None:
    x_values = [cities[i][0] for i in route]
    y_values = [cities[i][1] for i in route]

    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color="red", marker="o", linestyle="-", label="Ścieżka")
    plt.scatter(x_values, y_values, color="blue", zorder=5, label="Miasta")

    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=12, ha="right")

    plt.title("Optymalna ścieżka")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    N = 15
    depth = 2
    cities = generate_cities(N, radius=10, generate_inside=False)
    dist_mat = distance_matrix(cities)

    visualize_cities(cities)

    final_res, final_path = branch_and_bound_tsp(cities, N)

    visualize_best_path(cities, final_path)
    # final_res, best_path = branch_and_bound_tsp_parallel(cities, N, depth)
    # print("Minimum cost:", final_res)
    # print("Path taken:", best_path)
