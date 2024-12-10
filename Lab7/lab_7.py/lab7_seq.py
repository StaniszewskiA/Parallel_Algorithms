import numpy as np
import pandas as pd

G = 6.67430e-11  
EPS = 1e-10  
MASS_SCALAR = 1e30 

def compute_forces(stars: list[dict]) -> list[dict]:
    for star in stars:
        star['force'] = np.zeros(3)  
        star['acceleration'] = np.zeros(3)  

    for i, star_i in enumerate(stars):
        for j, star_j in enumerate(stars):
            if i != j: 
                r_vec = star_j['position'] - star_i['position']
                r_mag = np.linalg.norm(r_vec)
                
                force = G * star_i['mass'] * star_j['mass'] * r_vec / (r_mag**3 + EPS)
                star_i['force'] += force

        star_i['acceleration'] = star_i['force'] / star_i['mass']

    return stars

if __name__ == "__main__":
    np.random.seed(42)
    
    stars = [
        {'position': np.array([0.0, 0.0, 0.0]), 'mass': MASS_SCALAR},
        {'position': np.array([1.0, 2.0, 3.0]), 'mass': MASS_SCALAR},
        {'position': np.array([2.0, 4.0, 6.0]), 'mass': MASS_SCALAR},
        {'position': np.array([3.0, 6.0, 9.0]), 'mass': MASS_SCALAR},
        {'position': np.array([4.0, 8.0, 12.0]), 'mass': MASS_SCALAR},
    ]
    
    stars = compute_forces(stars)
    
    data = {
        "Pozycja (x, y, z)": [star['position'] for star in stars],
        "Masa": [star['mass'] for star in stars],
        "Siła (Fx, Fy, Fz)": [star['force'] for star in stars],
        "Przyspieszenie (Ax, Ay, Az)": [star['acceleration'] for star in stars],
    }
    
    df = pd.DataFrame(data)
    pd.set_option('display.float_format', lambda x: f'{x:.2e}')  
    print(df)
