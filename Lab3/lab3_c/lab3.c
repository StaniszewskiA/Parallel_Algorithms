#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void sieve_of_eratosthenes(int limit, int **primes, int *prime_count) {
    int *sieve = (int *)malloc((limit + 1) * sizeof(int));  
    if (sieve == NULL) {
        printf("Memory allocation failed\n");
        return;  
    }

    for (int i = 0; i <= limit; i++) {
        sieve[i] = 1;
    }
    sieve[0] = sieve[1] = 0;


    for (int num = 2; num <= sqrt(limit); num++) {
        if (sieve[num] == 1) {
            for (int multiple = num * num; multiple <= limit; multiple += num) {
                sieve[multiple] = 0;
            }
        }
    }

    *prime_count = 0;
    for (int i = 2; i <= limit; i++) {
        if (sieve[i]) {
            (*prime_count)++;
        }
    }

    *primes = (int *)malloc(*prime_count * sizeof(int));
    if (*primes == NULL) {
        printf("Memory allocation for primes failed\n");
        free(sieve);
        return;
    }

    int index = 0;
    for (int i = 2; i <= limit; i++) {
        if (sieve[i]) {
            (*primes)[index++] = i;
        }
    }

    free(sieve);
}

int main() {
    int limit = 99999;  
    double total_time = 0;   

    clock_t start_time = clock();  

    int *primes = NULL;
    int prime_count = 0;
    sieve_of_eratosthenes(limit, &primes, &prime_count); 

    clock_t end_time = clock(); 
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;  
    total_time += elapsed_time;  

    printf("Found %d primes, elapsed time: %.2f seconds.\n",prime_count, elapsed_time);


    free(primes);  

    return 0;
}
