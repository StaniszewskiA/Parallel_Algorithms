#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define Nx 20
#define Ny 20

int main() {
    double Lx = 10.0;
    double Ly = 10.0;
    double g = 1.0;
    double lambda_ = 1.0;
    double dx = Lx / (Nx - 1);
    double dy = Ly / (Ny - 1);
    double tolerance = 1e-6;
    double error = 1.0;

    double T[Ny][Nx] = {0};

    // Warunki brzegowe
    for (int i = 0; i < Nx; i++) {
        T[0][i] = 0.0;
        T[Ny-1][i] = 0.0;
    }
    for (int j = 0; j < Ny; j++) {
        T[j][0] = 0.0;
        T[j][Nx-1] = 0.0;
    }

    double T_old[Ny][Nx] = {0};

    while (error > tolerance) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                T_old[j][i] = T[j][i];
            }
        }

        for (int j = 1; j < Ny - 1; j++) {
            for (int i = 1; i < Nx - 1; i++) {
                T[j][i] = (T_old[j + 1][i] + T_old[j - 1][i] + 
                           T_old[j][i + 1] + T_old[j][i - 1] - 
                           (g / lambda_) * (dx * dx + dy * dy)) / 4.0;
            }
        }

        error = 0.0;
        for (int j = 1; j < Ny - 1; j++) {
            for (int i = 1; i < Nx - 1; i++) {
                double diff = fabs(T[j][i] - T_old[j][i]);
                if (diff > error) {
                    error = diff;
                }
            }
        }
    }

    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            printf("%6.2f ", T[j][i]);
        }
        printf("\n");
    }

    return 0;
}