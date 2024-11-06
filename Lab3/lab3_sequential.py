import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    Lx: float = 10.0
    Ly: float = 10.0
    Nx: int = 100
    Ny: int = 100
    g: float = 1.0
    lambda_: float = 1.0

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    T = np.zeros((Ny, Nx))

    # Warunki brzegowe
    T[:, 0] = 0
    T[:, -1] = 0
    T[0, :] = 0
    T[-1, :] = 0

    tolerance: float = 1e-6
    error: float = 1.0

    while error > tolerance:
        T_old = T.copy()

        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                T[j, i] = (T_old[j + 1, i] + T_old[j - 1, i] + 
                        T_old[j, i + 1] + T_old[j, i - 1] - 
                        (g / lambda_) * (dx ** 2 + dy ** 2)) / 4.0
                
        error: float = np.max(np.abs(T - T_old))

    plt.imshow(T, cmap='hot', origin='lower', extent=[0, Lx, 0, Ly])
    plt.colorbar(label="Temperature")
    plt.title("2D Heat Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print(f"{(end-start):.7}")