import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def main():
    Lx: float = 10.0
    Ly: float = 10.0
    Nx: int = 10
    Ny: int = 10
    g: float = 1.0
    lambda_: float = 1.0

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rows_per_process = Ny // size
    extra_rows = Ny % size

    if rank < extra_rows:
        local_rows = rows_per_process + 1
    else:
        local_rows = rows_per_process

    T_local = np.zeros((local_rows, Nx))
    T_local[0, :] = 0
    T_local[-1, :] = 0

    tolerance: float = 1e-6
    error: float = 1.0

    while error > tolerance:
        T_old = T_local.copy()

        for j in range(1, local_rows - 1):
            for i in range(1, Nx - 1):
                T_local[j, i] = (T_old[j + 1, i] + T_old[j - 1, i] +
                                 T_old[j, i + 1] + T_old[j, i - 1] - 
                                 (g / lambda_) * (dx ** 2 + dy ** 2)) / 4.0

        if local_rows > 1:
            if rank > 0:
                comm.Send(T_local[1, :], dest=rank - 1, tag=0)
                comm.Recv(T_local[0, :], source=rank - 1, tag=1)

            if rank < size - 1:
                comm.Send(T_local[-2, :], dest=rank + 1, tag=1)
                comm.Recv(T_local[-1, :], source=rank + 1, tag=0)

        error = np.max(np.abs(T_local - T_old))
        error = comm.allreduce(error, op=MPI.MAX)

    T_local_flat = T_local.flatten()
    counts = np.array(comm.gather(len(T_local_flat), root=0))

    if rank == 0:
        T_global_flat = np.empty(sum(counts), dtype=T_local.dtype)
    else:
        T_global_flat = None

    comm.Gatherv(T_local_flat, (T_global_flat, counts), root=0)

    if rank == 0:
        T_global = np.zeros((Ny, Nx))
        row_offset = 0
        for i in range(size):
            local_rows_i = rows_per_process + 1 if i < extra_rows else rows_per_process
            T_global[row_offset:row_offset + local_rows_i, :] = \
                T_global_flat[row_offset * Nx:(row_offset + local_rows_i) * Nx].reshape(local_rows_i, Nx)
            row_offset += local_rows_i
        return T_global
    return None

if __name__ == "__main__":
    start = time.time()
    T_global = main()
    end = time.time()
    print(f"{(end - start):.7}")

    # if T_global is not None:
    #     plt.imshow(T_global, cmap='hot', origin='lower', extent=[0, 10, 0, 10])
    #     plt.colorbar(label="Temperature")
    #     plt.title("2D Heat Distribution")
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.show()
    #     plt.savefig("heatmap.png")
