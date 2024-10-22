import math
import numpy as np
import time

def sieve_of_erathosthenes(limit: int) -> list[int]:
    sieve: list[bool] = np.ones(limit + 1, dtype=bool)
    sieve[0] = False
    sieve[1] = False

    for num in range(2, int(math.sqrt(limit)) + 1):
        if sieve[num]:
            sieve[num*num:limit+1:num] = False

    return np.nonzero(sieve)[0]

if __name__ == "__main__":
    limit: int = 999999999
    total_time: float = 0
    iterations: int = 5

    for i in range(iterations):
        start_time = time.time()
        result = sieve_of_erathosthenes(limit)
        end_time = time.time()

        elapsed_time: float = end_time - start_time
        total_time += elapsed_time

        print(f"Iteration {i + 1}: Found {len(result)} primes, elapsed time: {elapsed_time:.2f} seconds.")

    average_time = total_time / iterations
    print(f"Average time over {iterations} iterations: {average_time:.2f} seconds.")