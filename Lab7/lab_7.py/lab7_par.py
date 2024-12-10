from mpi4py import MPI
import numpy as np

G = 6.67430e-11  
EPS = 1e-10
MASS_SCALAR = 1e30

stars = [
    {'position': np.array([0.0, 0.0, 0.0]), 'mass': MASS_SCALAR},
    {'position': np.array([1.0, 2.0, 3.0]), 'mass': MASS_SCALAR},
    {'position': np.array([2.0, 4.0, 6.0]), 'mass': MASS_SCALAR},
    {'position': np.array([3.0, 6.0, 9.0]), 'mass': MASS_SCALAR},
    {'position': np.array([4.0, 8.0, 12.0]), 'mass': MASS_SCALAR}
]

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
        star['velocity'] = np.zeros(3)

    comm = MPI.COMM_WORLD

    for step in range(time_steps):
        local_acc = task_2(stars, rank, size)

        # Aktualizacja pozycji
        for i, star in enumerate(local_stars): 
            star['position'] += star['velocity'] * dt + 0.5 * local_acc[i] * dt**2

        new_local_acc = task_2(stars, rank, size)

        # Aktualizacja prędkości
        for i, star in enumerate(local_stars):
            star['velocity'] += 0.5 * (local_acc[i] + new_local_acc[i]) * dt

        # Synchronizacja
        all_stars = comm.allgather(local_stars)
        stars = [star for sublist in all_stars for star in sublist]

    return local_stars

def main():
    task = 3

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if task == 1:
        accs_task_1 = task_1(stars, rank, size)
        accs_task_2 = task_2(stars, rank, size)

        all_accs_task_1 = comm.gather(accs_task_1, root=0)
        all_accs_task_2 = comm.gather(accs_task_2, root=0)

        if rank == 0:
            final_accs_task_1 = np.concatenate(all_accs_task_1, axis=0)
            final_accs_task_2 = np.concatenate(all_accs_task_2, axis=0)

            print("Przyspieszenia (Task 1):")
            for i, acc in enumerate(final_accs_task_1):
                print(f"Gwiazda {i}: przyspieszenie = {acc}")

            print("\nPrzyspieszenia (Task 2):")
            for i, acc in enumerate(final_accs_task_2):
                print(f"Gwiazda {i}: przyspieszenie = {acc}")
    else:
        time_steps = 100
        dt = 1e5

        trajectories = task_3(stars, rank, size, time_steps, dt)
        all_trajectories = comm.gather(trajectories, root=0)

        if rank == 0:
            final_trajectories = [star for sublist in all_trajectories for star in sublist]

            print("\nTrajektorie (Task 3):")
            for i, star in enumerate(final_trajectories):
                print(f"Gwiazda {i}: pozycja = {star['position']}, prędkość = {star['velocity']}")

if __name__ == "__main__":
    main()