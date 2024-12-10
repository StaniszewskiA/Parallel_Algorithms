from mpi4py import MPI
import numpy as np
import pandas as pd
import time

G = 6.67430e-11  
EPS = 1e-10
MASS_SCALAR = 1e30
STARS = 1000

def generate_stars(num_stars):
    stars = []
    for _ in range(num_stars):
        position = np.random.rand(2) * 10  
        mass = MASS_SCALAR 
        stars.append({'position': position, 'mass': mass})
    return stars

def task_1(stars, rank, size):
    N = len(stars)
    stars_per_process = N // size
    local_stars = stars[rank * stars_per_process: (rank + 1) * stars_per_process]

    transfer_buffer = np.copy(local_stars)
    local_acc = np.zeros_like(local_stars)

    comm = MPI.COMM_WORLD

    for _ in range(size - 1):
        left_rank = (rank - 1) % size
        right_rank = (rank + 1) % size

        comm.send(transfer_buffer, dest=left_rank)
        transfer_buffer = comm.recv(source=right_rank)

        for i, star_i in enumerate(local_stars):
            m_i = star_i['mass']
            r_i = star_i['position']

            for star_j  in transfer_buffer:
                r_j = star_j['position']
                m_j = star_j['mass']

                r_vec = r_j - r_i
                r_mag = np.linalg.norm(r_vec)
                force = G * m_i * m_j * r_vec / (r_mag**3 + EPS)
                local_acc[i] += force / m_i

    return local_acc

def task_2(stars, rank, size):
    N = len(stars)
    stars_per_process = N // size
    local_stars = stars[rank * stars_per_process: (rank + 1) * stars_per_process]

    transfer_buffer = np.copy(local_stars)
    local_acc = np.zeros_like(local_stars)

    comm = MPI.COMM_WORLD

    for _ in range(size // 2):
        left_rank = (rank - 1) % size
        right_rank = (rank + 1) % size

        comm.send(transfer_buffer, dest=left_rank)
        transfer_buffer = comm.recv(source=right_rank)
        
        for i, star_i in enumerate(local_stars):
            m_i = star_i['mass']
            r_i = star_i['position']

            for star_j  in transfer_buffer:
                r_j = star_j['position']
                m_j = star_j['mass']

                r_vec = r_j - r_i
                r_mag = np.linalg.norm(r_vec)
                force = G * m_i * m_j * r_vec / (r_mag**3 + EPS)
                local_acc[i] += force / m_i

        transfer_buffer = local_stars.copy()

    return local_acc

def task_3(stars, rank, size, time_steps, dt):
    N = len(stars)
    stars_per_process = N // size
    local_stars = stars[rank * stars_per_process: (rank + 1) * stars_per_process]

    for star in local_stars:
        star['velocity'] = np.zeros(2)

    comm = MPI.COMM_WORLD

    for _ in range(time_steps):
        local_acc = task_2(stars, rank, size)

        for i, star in enumerate(local_stars):
            # Pozycja
            star['position'] += star['velocity'] * dt
            # Prędkość
            star['velocity'] += local_acc[i] * dt

        all_stars = comm.allgather(local_stars)
        stars = [star for sublist in all_stars for star in sublist]

    return local_stars

def main():
    pd.set_option('display.float_format', lambda x: f'{x:.2e}') 

    stars = generate_stars(STARS)

    task = 3

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if task == 1:
        start = time.time()
        accs_task_1 = task_1(stars, rank, size)
        all_accs_task_1 = comm.gather(accs_task_1, root=0)

        if rank == 0:
            final_accs_task_1 = np.concatenate(all_accs_task_1, axis=0)
            end = time.time()

            print(f"Elapsed time {end - start}")

            df_1 = pd.DataFrame(final_accs_task_1)
            df_1.to_csv("acc_par_1.csv", index=False)

    if task == 2:
        start = time.time()
        accs_task_2 = task_2(stars, rank, size)
        all_accs_task_2 = comm.gather(accs_task_2, root=0)

        if rank == 0:
            final_accs_task_2 = np.concatenate(all_accs_task_2, axis=0)
            end = time.time()

            print(f"Elapsed time {end - start}")

            df_2 = pd.DataFrame(final_accs_task_2)
            df_2.to_csv("acc_par_2.csv", index=False)

    if task == 3:
        time_steps = 100
        dt = 1e5

        start = time.time()
        trajectories = task_3(stars, rank, size, time_steps, dt)
        all_trajectories = comm.gather(trajectories, root=0)

        end = time.time()

        if rank == 0:
            final_trajectories = [star for sublist in all_trajectories for star in sublist]
 
            print(f"Elapsed time {end - start}")

            print(final_trajectories)

            df_3 = pd.DataFrame(final_trajectories)
            df_3.to_csv("traj_par.csv", index=False)

if __name__ == "__main__":
    main()