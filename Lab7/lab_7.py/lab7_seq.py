import numpy as np
import pandas as pd
import time

from tqdm import tqdm

G = 6.67430e-11  
EPS = 1e-10  
MASS_SCALAR = 1e30 
STARS = 1000
TIME_STEPS = 100 
DT = 1e5  

def generate_stars(num_stars):
    stars = []
    for _ in range(num_stars):
        position = np.random.rand(2) * 10  
        velocity = np.random.rand(2)  
        mass = MASS_SCALAR 
        stars.append({'position': position, 'velocity': velocity, 'mass': mass})
    return stars

def compute_forces(stars: list[dict]) -> list[dict]:
    for star in stars:
        star['force'] = np.zeros(2) 
        star['acceleration'] = np.zeros(2) 

    for i, star_i in tqdm(enumerate(stars)):
        for j, star_j in enumerate(stars):
            if i != j: 
                r_vec = star_j['position'] - star_i['position']
                r_mag = np.linalg.norm(r_vec) 
                
                force = G * star_i['mass'] * star_j['mass'] * r_vec / (r_mag**3 + EPS)
                star_i['force'] += force  

        star_i['acceleration'] = star_i['force'] / star_i['mass'] 

    return stars

def euler_integration(stars: list[dict], dt: float, time_steps: int) -> list[dict]:
    """
    Funkcja do integracji pozycji i prędkości gwiazd metodą Eulera
    """
    for _ in range(time_steps):
        stars = compute_forces(stars)

        for star in stars:
            # Aktualizacja pozycji: r(t+Δt) = r(t) + v(t) * Δt
            star['position'] += star['velocity'] * dt
            # Aktualizacja prędkości: v(t+Δt) = v(t) + a(t) * Δt
            star['velocity'] += star['acceleration'] * dt

    return stars

if __name__ == "__main__":
    np.random.seed(42)
    
    stars = generate_stars(STARS)

    start = time.time()
    stars = euler_integration(stars, DT, TIME_STEPS)  # Symulacja za pomocą metody Eulera
    end = time.time()

    print(f"Elapsed time: {end - start} seconds")

    # Zbieranie danych
    data = {
        "Pozycja (x, y)": [star['position'] for star in stars],
        "Prędkość (Vx, Vy)": [star['velocity'] for star in stars],
        "Masa": [star['mass'] for star in stars],
        "Siła (Fx, Fy)": [star['force'] for star in stars],
        "Przyspieszenie (Ax, Ay)": [star['acceleration'] for star in stars],
    }
    
    # Tworzenie DataFrame z wynikami
    df = pd.DataFrame(data)
    pd.set_option('display.float_format', lambda x: f'{x:.2e}')  # Ustawienie formatu wyświetlania liczb

    # Zapisanie danych do pliku CSV
    df.to_csv("stars_euler_integration.csv", index=False)
