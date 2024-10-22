from mpi4py import MPI
import numpy as np
import math

def sieve_of_erathosthenes(limit: int) -> list[int]:
    sieve: list[bool] = np.ones(limit + 1, dtype=bool)
    sieve[0] = False
    sieve[1] = False

    for num in range(2, int(math.sqrt(limit)) + 1):
        if sieve[num]:
            sieve[num*num:limit+1:num] = False

    return np.nonzero(sieve)[0]

def parallel_sieve(n):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    sqrt_n: int = int(math.sqrt(n))

    # Punkt 1
    primes_in_B: list[int] = sieve_of_erathosthenes(sqrt_n)

    # Punkt 2
    start_C: int = sqrt_n + 1 + rank * (n - sqrt_n) // size
    end_C: int = sqrt_n + 1 + (rank + 1) * (n - sqrt_n) // size - 1

    # Punkt 3
    sieve: list[bool] = np.ones(end_C - start_C + 1, dtype=bool)

    for prime in primes_in_B:
        first_multiple = max(prime*prime, (start_C + prime - 1) // prime*prime)
        sieve[first_multiple - start_C:end_C - start_C + 1:prime] = False

    local_primes = np.nonzero(sieve)[0] + start_C

    # Punkt 4
    all_primes = comm.gather(local_primes, root = 0)

    if rank == 0:
        all_primes = np.concatenate([primes_in_B] + all_primes)
        return all_primes
    else:
        return None

if __name__ == "__main__":
    limit: int = 100000
    result = parallel_sieve(limit)
    print(result)