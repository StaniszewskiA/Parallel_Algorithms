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

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine local rows for each process
    rows_per_process = Ny // size
    extra_rows = Ny % size
    local_rows = rows_per_process + 1 if rank < extra_rows else rows_per_process

    # Initialize the local temperature array
    T_local = np.empty((local_rows, Nx), dtype=np.float64)
    T_local[0, :] = 0
    T_local[-1, :] = 0

    tolerance: float = 1e-6
    error: float = 1.0

    while error > tolerance:
        T_old = T_local.copy()

        # Update temperature (skip boundary rows)
        for j in range(1, local_rows - 1):
            for i in range(1, Nx - 1):
                T_local[j, i] = (T_old[j + 1, i] + T_old[j - 1, i] +
                                 T_old[j, i + 1] + T_old[j, i - 1] -
                                 (g / lambda_) * (dx ** 2 + dy ** 2)) / 4.0

        # Communicate boundary rows with neighbors
        if local_rows > 1:
            if rank > 0:
                comm.Sendrecv(T_local[1, :], dest=rank - 1, sendtag=0,
                              recvbuf=T_local[0, :], source=rank - 1, recvtag=1)
            if rank < size - 1:
                comm.Sendrecv(T_local[-2, :], dest=rank + 1, sendtag=1,
                              recvbuf=T_local[-1, :], source=rank + 1, recvtag=0)

        # Calculate the maximum error across all processes
        local_error = np.max(np.abs(T_local - T_old))
        error = comm.allreduce(local_error, op=MPI.MAX)

    # Flatten local temperature data for gathering
    T_local_flat = T_local.flatten()

    # Initialize counts and displacements
    counts = np.zeros(size, dtype=int)
    displacements = np.zeros(size, dtype=int)

    # Gather the size of the local arrays from each rank
    comm.Gather(len(T_local_flat), counts, root=0)

    if rank == 0:
        # Compute the displacements for each rank
        displacements[1:] = np.cumsum(counts[:-1])

    T_global_flat = None
    if rank == 0:
        T_global_flat = np.empty(np.sum(counts), dtype=T_local.dtype)

    # Perform the gather operation
    comm.Gatherv(sendbuf=T_local_flat, recvbuf=(T_global_flat, counts, displacements), root=0)

    # Reshape and reconstruct global temperature grid on root
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
    MPI.COMM_WORLD.Barrier()  # Synchronize all processes before timing ends
    end = time.time()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Execution Time: {(end - start):.7} seconds")
