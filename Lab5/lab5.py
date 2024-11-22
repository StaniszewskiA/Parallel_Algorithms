import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def main():
    Lx: float = 10.0
    Ly: float = 10.0
    Nx: int = 64
    Ny: int = 64
    g: float = 1.0
    lambda_: float = 1.0
    omega: float = 1.8 

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rows_per_process = Ny // size
    extra_rows = Ny % size
    local_rows = rows_per_process + 1 if rank < extra_rows else rows_per_process

    T_local = np.zeros((local_rows + 2, Nx)) 
    T_local[0, :] = 0 
    T_local[-1, :] = 0  

    tolerance: float = 1e-6
    error: float = 1.0

    while error > tolerance:
        error_local = 0.0

        for j in range(1, local_rows + 1):
            for i in range(1, Nx - 1):
                T_new = (T_local[j + 1, i] + T_local[j - 1, i] +
                         T_local[j, i + 1] + T_local[j, i - 1] - 
                         (g / lambda_) * (dx ** 2 + dy ** 2)) / 4.0
                T_local[j, i] = (1 - omega) * T_local[j, i] + omega * T_new
                error_local = max(error_local, abs(T_local[j, i] - T_new))

        reqs = []
        if rank > 0:
            reqs.append(comm.Irecv(T_local[0, :], source=rank - 1, tag=0))
            reqs.append(comm.Isend(T_local[1, :], dest=rank - 1, tag=1))
        if rank < size - 1:
            reqs.append(comm.Irecv(T_local[-1, :], source=rank + 1, tag=1))
            reqs.append(comm.Isend(T_local[-2, :], dest=rank + 1, tag=0))

        MPI.Request.Waitall(reqs)

        error = comm.allreduce(error_local, op=MPI.MAX)

    T_local_flat = T_local[1:-1, :].flatten() 
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
    print(f"Execution Time: {(end - start):.7f} seconds")