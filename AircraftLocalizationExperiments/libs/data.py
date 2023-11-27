import numpy as np
import pandas as pd
from libs import geo
from typing import Dict

TEST_DIR = 'data'

def get_sensors(fixed=False):
    if fixed:
        return pd.read_csv(f'{TEST_DIR}/sensors_fixed.csv')
    return pd.read_csv(f'{TEST_DIR}/sensors.csv')


def get_sensor_coords_cartesian(fixed=False) -> Dict[int, np.ndarray]:
    sensors = get_sensors(fixed=fixed)
    id = sensors['serial'].values
    coords = sensors[['latitude', 'longitude', 'height']].values
    coords_cartesian = [geo.latlon_to_cartesian(*c) for c in coords]
    return dict(zip(id, coords_cartesian))

def get_test_dataset():
    return pd.read_csv(f'{TEST_DIR}/round1_competition.csv')