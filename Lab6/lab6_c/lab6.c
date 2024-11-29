#include "lab6.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


void generate_cities(
    City *cities, 
    int N, 
    double radius, 
    int generate_inside,
    double noise_level
    ) {
        for (int i = 0; i < N; i++) {
            double angle = 2 * M_PI * i / N + ((rand() / (double)RAND_MAX) * noise_level - noise_level / 2);
            double rad = generate_inside ? (rand() / (double)RAND_MAX) * radius : radius;
            cities[i].x = rad * cos(angle);
            cities[i].y = rad * sin(angle);
        }
    }

double **allocate_distance_matrix(int N) {
    double **matrix = malloc(N * sizeof(double *));
    for (int i = 0; i < N; i ++) {
        matrix[i] = malloc(N * sizeof(double));
    }

    return matrix;
}

void free_distance_matrix(double **matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void calculate_distance_matrix(City *cities, double **dist_mat, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double dx = cities[i].x - cities[j].x;
            double dy = cities[i].y - cities[j].y;
            dist_mat[i][j] = sqrt(dx * dx + dy * dy);
        }
    }
}

void copy_to_final(int *curr_path, int *final_path, int N) {
    for (int i = 0; i < N; i++) {
        final_path[i] = curr_path[i];
    }
    final_path[N] = curr_path[0];
}

double first_min(double **adj, int i, int N) {
    double min = MAXSIZE;
    for (int j = 0; j < N; j++) {
        if (adj[i][j] < min && i != j) {
            min = adj[i][j];
        }
    }
    return min;
}

double second_min(double **adj, int i, int N) {
    double first = MAXSIZE, second = MAXSIZE;
    for (int j = 0; j < N; j++) {
        if (i == j) continue;
        if (adj[i][j] <= first) {
            second = first;
            first = adj[i][j];
        } else if (adj[i][j] < second) {
            second = adj[i][j];
        }
    }

    return second;
}

void tsp_recursive(double **adj, double curr_bound, double curr_weight, int level, 
                   int *curr_path, int *visited, int *final_path, int N, double *final_res) {
    if (level == N) {
        if (adj[curr_path[level - 1]][curr_path[0]] != 0) {
            double curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]];
            if (curr_res < *final_res) {
                copy_to_final(curr_path, final_path, N);
                *final_res = curr_res;
            }
        }
        return;
    }

    for (int i = 0; i < N; i++) {
        if (adj[curr_path[level - 1]][i] != 0 && !visited[i]) {
            double temp = curr_weight + adj[curr_path[level - 1]][i];
            double bound_temp = curr_bound;

            if (level == 1) {
                bound_temp -= (first_min(adj, curr_path[level - 1], N) + first_min(adj, i, N)) / 2;
            } else {
                bound_temp -= (second_min(adj, curr_path[level - 1], N) + first_min(adj, i, N)) / 2;
            }

            if (temp + bound_temp < *final_res) {
                curr_path[level] = i;
                visited[i] = 1;
                tsp_recursive(adj, bound_temp, temp, level + 1, curr_path, visited, final_path, N, final_res);
            }

            visited[i] = 0;
        }
    }
}

void branch_and_bound_tsp(City *cities, int N, int *final_path, double *final_res) {
    double **dist_mat = allocate_distance_matrix(N);
    calculate_distance_matrix(cities, dist_mat, N);

    int curr_path[N + 1], visited[N];
    for (int i = 0; i < N; i++) visited[i] = 0;

    double curr_bound = 0;
    for (int i = 0; i < N; i++) {
        curr_bound += (first_min(dist_mat, i, N) + second_min(dist_mat, i, N));
    }

    curr_bound = ceil(curr_bound / 2);

    curr_path[0] = 0;
    visited[0] = 1;

    *final_res = MAXSIZE;
    tsp_recursive(dist_mat, curr_bound, 0, 1, curr_path, visited, final_path, N, final_res);

    free_distance_matrix(dist_mat, N);
}

int main() {
    int N = 10;
    City cities[N];
    int final_path[N + 1];
    double final_res;

    generate_cities(cities, N, 10.0, 0, 0.1);
    branch_and_bound_tsp(cities, N, final_path, &final_res);

    printf("Minimum cost: %f\n", final_res);
    printf("Path: ");
    for (int i = 0; i <= N; i++) {
        printf("%d ", final_path[i]);
    }
    printf("\n");

    return 0;
}