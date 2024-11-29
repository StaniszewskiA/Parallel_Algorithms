#ifndef lab6_h
#define lab6_h

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#define MAXSIZE DBL_MAX

typedef struct {
    double x;
    double y;
} City;

void generate_cities(City *cities, int N, double radius, int generate_inside, double noise_level);
double **allocate_distance_matrix(int N);
void free_distance_matrix(double **matrix, int N);
void calculate_distance_matrix(City *cities, double **dist_mat, int N);
void copy_to_final(int *curr_path, int *final_path, int N);
double first_min(double **adj, int i, int N);
double second_min(double **adj, int i, int N);
void tsp_recursive(double **adj, double curr_bound, double curr_weight, int level, 
                   int *curr_path, int *visited, int *final_path, int N, double *final_res);
void branch_and_bound_tsp(City *cities, int N, int *final_path, double *final_res);
void visualize_cities(City *cities, int N); 
void visualize_best_path(City *cities, int N, int *route); 

#endif 