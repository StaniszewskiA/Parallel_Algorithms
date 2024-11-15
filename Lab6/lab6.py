import numpy as np

def generate_cities(N: int, radius: float = 1.0) -> list[tuple[float, float]]:
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cities = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    
    return cities

if __name__ == "__main__":
    N = 6
    d = 2

    result = generate_cities(N)