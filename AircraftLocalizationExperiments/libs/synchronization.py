import numpy as np
import pandas as pd
from typing import Dict
from tqdm import tqdm

from libs import mlat_3points
from libs import geo
import math
from sklearn.linear_model import LinearRegression


def synchronize_sensors(train_df: pd.DataFrame, sensor_coords: Dict[int, np.ndarray], params: Dict[str, bool]) -> np.ndarray:
    diffs = _get_diffs(train_df, sensor_coords)


    num_sensors = np.max(list(sensor_coords.keys()))
    MIN_DIFFS_COUNT = 400
    A = []
    B = []

    for k, v in tqdm(diffs.items()):
        if len(v) < MIN_DIFFS_COUNT:
            continue

        id1, id2 = k
        row = np.zeros(num_sensors + 1, dtype=np.float32)
        row[id1] = 1
        row[id2] = -1
        A.append(row)
        B.append(np.median(v))

    A = np.array(A)
    B = np.array(B)

    # print(A.shape, B.shape)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(A, B)
    shift = lr.coef_
    # print(shift.shape)

    # print(shift)
    # print(len(np.where(np.abs(shift) > 1e-15)[0]))
    # print(np.where(np.abs(shift) > 1e-15))
    # print(shift[np.where(np.abs(shift) > 1e-15)])

    # score = math.sqrt(np.mean(np.square(A @ shift - B))) * geo.LIGHT_SPEED
    # print('score', score)

    return shift


def _get_diffs(train_df: pd.DataFrame, sensor_coords: Dict[int, np.ndarray]) -> Dict:
    diffs = dict()

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc='Sensor synchronization pre-calculation'):
        meas = row['measurements']

        receiver_coords = []
        receiver_time = []
        receiver_ids = []

        for m in meas:
            sensor_id, timestamp, power = m

            receiver_coords.append(sensor_coords[sensor_id])
            receiver_time.append(timestamp)
            receiver_ids.append(sensor_id)

        if len(receiver_coords) < 2:
            continue

        receiver_ids = np.array(receiver_ids)
        receiver_coords = np.array(receiver_coords)
        receiver_time = np.array(receiver_time) / 1e9
        receiver_time -= receiver_time.min()

        if receiver_time.max() > 400000 / geo.LIGHT_SPEED:
            continue

        lat = row['latitude']
        long = row['longitude']
        geoAltitude = row['geoAltitude']

        TDoA = mlat_3points._get_TDoA(receiver_coords, (lat, long, geoAltitude), normalize=False)
        if TDoA.max() > 400000 / geo.LIGHT_SPEED:
            continue

        TDoA -= TDoA.min()

        n = len(receiver_ids)
        for i in range(n):
            for j in range(i + 1, n):
                id1 = receiver_ids[i]
                id2 = receiver_ids[j]

                v = (receiver_time[i] - receiver_time[j]) - (TDoA[i] - TDoA[j])

                if id2 < id1:
                    id1, id2 = id2, id1
                    v = -v

                if (id1, id2) not in diffs:
                    diffs[(id1, id2)] = [v]
                else:
                    diffs[(id1, id2)].append(v)

    return diffs