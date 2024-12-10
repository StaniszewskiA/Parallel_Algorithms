import numpy as np
import matplotlib.pyplot as plt

problem_sizes = ["d=2"]
T1 = np.array([779.56])
T2 = np.array([244.64])
T4 = np.array([183.43])
T8 = np.array([157.95])

P_values = np.array([1, 2, 4, 8])

S = {}
E = {}
f = {}

for i, size in enumerate(problem_sizes):
    S[size] = [1.0]  
    E[size] = [1.0]  

    for j, P in enumerate(P_values[1:], start=1):
        T_P = T2[i] if P == 2 else T4[i]
        S[size].append(T1[i] / T_P)
        E[size].append(S[size][-1] / P)

    f[size] = [0.0]  
    for j, P in enumerate(P_values[1:], start=1):
        T_P = T2[i] if P == 2 else T4[i]
        f[size].append((T1[i] / T_P - 1) * (P / (P - 1)))

plt.figure(figsize=(12, 8))

plt.figure()
for size in problem_sizes:
    plt.plot(P_values, S[size], marker='o', label=f'Ilość procesów: {size}')
plt.title('Wykres przyspieszenia S(P)')
plt.xlabel('Liczba procesów (P)')
plt.ylabel('Speedup (S(P))')
plt.legend()
plt.grid()
plt.show()

plt.figure()
for size in problem_sizes:
    plt.plot(P_values, E[size], marker='o', label=f'Ilość procesów: {size}')
plt.title('Wykres efektywności E(P)')
plt.xlabel('Liczba procesów (P)')
plt.ylabel('Efektywność (E(P))')
plt.legend()
plt.grid()
plt.show()

plt.figure()
for size in problem_sizes:
    plt.plot(P_values[1:], f[size][1:], marker='o', label=f'Ilość procesów: {size}')
plt.title('Metryka Karpa-Flatta f(P)')
plt.xlabel('Liczba procesów (P)')
plt.ylabel('Metryka Karpa-Flata (f(P))')
plt.legend()
plt.grid()
plt.show()