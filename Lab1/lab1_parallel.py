import mpi4py
import numpy as np

from mpi4py import MPI

def main():
    Lx: float = 10.0
    Ly: float = 10.0
    Nx: int = 20
    Ny: int = 20
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
        start_row = rank * local_rows
    else:
        local_rows = rows_per_process
        start_row = rank * local_rows + extra_rows

    end_row = start_row + local_rows

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
                
        if rank > 0: 
            comm.Send(T_local[1, :], dest=rank - 1, tag=0)  
            comm.Recv(T_local[0, :], source=rank - 1, tag=1)  

        if rank < size - 1:  
            comm.Send(T_local[-2, :], dest=rank + 1, tag=1)  
            comm.Recv(T_local[-1, :], source=rank + 1, tag=0)

        error = np.max(np.abs(T_local - T_old))

    T_global = None

    if rank == 0:
        T_global == np.empty((Ny, Nx))

    comm.Gather(T_local, T_global, root=0)

    if rank == 0:
        print(T_global)

if __name__ == "__name__":
    main()