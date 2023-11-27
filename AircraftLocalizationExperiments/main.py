import pandas as pd
import numpy as np
from geopy.distance import distance
import math

from libs import data
from libs import preprocessing
from libs import solver
from libs import geo
from libs import synchronization
from tqdm import tqdm

from typing import Dict, Optional

import matplotlib.pyplot as plt
import os


def solve_one_aircraft(aircraft_data: pd.DataFrame, sensor_coords, params: dict = dict(), sensor_shift: Optional[Dict[int, np.ndarray]] = None) -> pd.DataFrame:
    res_ids = None
    res_latlon = None

    meas_time, meas_sensor_coords, meas_received_time, meas_altitude, meas_ids = solver.parse_measurements(aircraft_data, sensor_coords, shift=sensor_shift)

    # calculate coordinates with Bancroft method for all points with 4 or more measurements
    mlat_4meas_time, mlat_4meas_coords, mlat_4meas_ids = solver.mlat_4meas_points(meas_time, meas_sensor_coords, meas_received_time, meas_ids)
    # convert 3D cartesian coordinates to WGS84
    mlat_4meas_coords_latlon = np.array([geo.cartesian_to_latlon(*x) for x in mlat_4meas_coords])

    if params['mlat_median_filtering']:
        mlat_4meas_time, mlat_4meas_coords_latlon, mlat_4meas_ids = solver.median_filtering(mlat_4meas_time, mlat_4meas_coords_latlon, mlat_4meas_ids)

    res_ids = mlat_4meas_ids
    res_latlon = mlat_4meas_coords_latlon

    if (params['interpolate_between_mlat']):
        interpolated_coords_latlon, interpolated_time = solver.interpolate_coords(mlat_4meas_time, mlat_4meas_coords_latlon, meas_time)
        res_ids = meas_ids
        res_latlon = interpolated_coords_latlon

    return pd.DataFrame({'id': res_ids,
                         'latitude': res_latlon[:, 0],
                         'longitude': res_latlon[:, 1],
                         'geoAltitude': 0})



def solve(params=None) -> pd.DataFrame:
    if params is None:
        params = dict()
    sensor_coords = data.get_sensor_coords_cartesian()

    all_df = data.get_test_dataset().copy()

    sensor_shifts = None

    if params['synchronize_sensors']:
        train_df = preprocessing.get_train_part(all_df)
        train_df = preprocessing.parse_measurements(train_df, filter_bad_measurements=params['filter_bad_measurements'], desc='Load train data')
        sensor_shifts = synchronization.synchronize_sensors(train_df, sensor_coords, params=params)

    test_df = preprocessing.get_test_part(all_df)
    test_df = preprocessing.parse_measurements(test_df, filter_bad_measurements=params['filter_bad_measurements'], desc='Load test data')

    res = []
    for aircraft_id, aircraft_data in tqdm(test_df.groupby('aircraft'), desc='Aircraft localization'):
        aircraft_track = solve_one_aircraft(aircraft_data, sensor_coords=sensor_coords, sensor_shift=sensor_shifts, params=params)
        aircraft_track['aircraft'] = aircraft_id
        res.append(aircraft_track)

    res = pd.concat(res)
    print('located:', res.shape, 'total:', test_df.shape)
    return res

def rmse_score(x):
    if len(x)==0:
        return 0

    return math.sqrt(np.sum(np.square(x))/len(x))
def calc_score(pred: pd.DataFrame):
    gt = pd.read_csv('data/round1_ground_truth.csv')
    gt_pred = gt.merge(pred, on='id', how='outer', suffixes=('_gt', '_pred'))

    coverage = np.sum(np.isfinite(gt_pred['latitude_pred']))/len(gt_pred)
    print(f'Coverage = {coverage*100:.02f} %')

    distances = []

    for _, row in gt_pred.iterrows():
        gt_pos = (row['latitude_gt'], row['longitude_gt'])
        pred_pos = (row['latitude_pred'], row['longitude_pred'])

        if np.isfinite(pred_pos[0]) and np.isfinite(pred_pos[1]):
            dist = distance(gt_pos, pred_pos).m
            distances.append(dist)

    rmse = rmse_score(distances)
    mean = np.mean(distances)
    median = np.median(distances)

    print(f'RMSE = {rmse:.02f}')
    print(f'mean = {mean:.02f}')
    print(f'median = {median:.02f}')

    k = int( round(len(distances)*0.9) )
    distances_top90 = np.sort(distances)[0:k]
    rmse_top90 = rmse_score(distances_top90)
    print(f'RMSE top90 = {rmse_top90:.02f}')

    # plt.hist(distances, bins=200)
    # plt.show()
    #
    # plt.hist(distances_top90, bins=200)
    # plt.show()

def save_result(res: pd.DataFrame, params: Dict):
    out_path = 'result'
    os.makedirs(out_path, exist_ok=True)

    filename = out_path + '/result_' + '_'.join([f'{k}={v}' for k, v in params.items()]) + '.csv'
    res.to_csv(filename, index=False)

if __name__ == '__main__':
    for synchronize_sensors in [False, True]:
        for mlat_median_filtering in [False, True]:
            for interpolate_between_mlat in [False, True]:
                params = {
                    'filter_bad_measurements': False,
                    'interpolate_between_mlat': interpolate_between_mlat,
                    'synchronize_sensors': synchronize_sensors,
                    'mlat_median_filtering': mlat_median_filtering
                }

                print(f'{synchronize_sensors= }')
                print(f'{mlat_median_filtering= }')
                print(f'{interpolate_between_mlat= }')

                res = solve(params=params)
                calc_score(res)
                save_result(res, params)
                print('-------------------------------')