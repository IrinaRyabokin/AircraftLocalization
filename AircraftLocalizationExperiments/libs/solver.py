from pyproj import geod
import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, Optional, List, Tuple

from libs import mlat_bancroft

def parse_measurements(df: pd.DataFrame,
                       sensor_coords: Dict[int, np.ndarray],
                       shift: Optional[Dict[int, float]] = None) -> Tuple[np.ndarray, List, List, np.ndarray, np.ndarray]:
    """
    :param df: test dataset for aircraft
    :param sensor_coords:
    :param shift: time shift for each sensor
    :return: filtered test dataset
    """
    meas_time = []
    meas_received_time = []
    meas_sensor_coords = []
    meas_altitude = []
    ids = []

    iterator = df.iterrows()
    for _, row in iterator:
        meas = row['measurements']

        receiver_coords = []
        receiver_time = []
        receiver_power = []

        curr_t = -1
        for m in meas:
            sensor_id, timestamp, power = m
            receiver_coords.append(sensor_coords[sensor_id])
            rt = timestamp / 1e9
            if shift is not None:
                rt = rt - shift[sensor_id]
            receiver_time.append(rt)
            receiver_power.append(power)

            if curr_t == -1:
                curr_t = receiver_time[0]

        # assert curr_t != -1
        if curr_t == -1:
            curr_t = row['timeAtServer']

        meas_time.append(curr_t)
        meas_received_time.append(np.array(receiver_time))
        meas_sensor_coords.append(np.array(receiver_coords))
        meas_altitude.append(row['baroAltitude'])
        ids.append(row['id'])

    return np.array(meas_time), meas_sensor_coords, meas_received_time, np.array(meas_altitude), np.array(ids)

def mlat_4meas_points(meas_time, meas_sensor_coords, meas_received_time, meas_ids):
    """
    Apply a Bancroft method for one point
    :param meas_time: server time
    :param meas_sensor_coords: sensor coords
    :param meas_received_time: receiving time for all sensors including shift time
    :return:
    """
    mlat_coord = []
    mlat_time = []
    mlat_ids = []
    for i in range(len(meas_time)):
        receiver_coords = meas_sensor_coords[i]
        receiver_time = meas_received_time[i]
        receiver_power = []

        if len(receiver_coords) < 4:
            continue

        receiver_time -= np.min(receiver_time)

        approx_coords_cartesian = mlat_bancroft.calc(receiver_coords, receiver_time, receiver_power)

        if approx_coords_cartesian is not None:
            mlat_coord.append(approx_coords_cartesian)
            mlat_time.append(meas_time[i])
            mlat_ids.append(meas_ids[i])

    return np.array(mlat_time), np.array(mlat_coord), np.array(mlat_ids)

def interpolate_coords(x0, y0, x1):
    if len(x0) < 1:
        res = np.empty((len(x1), 3))
        res[:] = np.NaN
        return res, x1

    if len(x0) == 1:
        res = np.empty((len(x1), 3))
        res[:] = np.NaN

        for i in range(len(x1)):
            if abs(x1[i] - x0[0]) < 1e-3:
                res[i] = y0[0]

        return res, x1

    x0 = np.array(x0)
    y0 = np.array(y0)
    x1 = np.array(x1)

    x0, x0_ind = np.unique(x0,return_index=True)
    y0 = y0[x0_ind]

    time_idx = 0
    time_left, time_right = x0[time_idx], x0[time_idx + 1]

    res = np.empty((len(x1), 3))
    res[:] = np.NaN

    for i in range(len(x1)):
        time = x1[i]

        if time < time_left:
            continue
        if time > time_right:
            time_idx += 1
            if time_idx + 1 >= len(x0):
                continue
            time_left, time_right = x0[time_idx], x0[time_idx + 1]

        if (time_left <= time) and (time <= time_right):
            p_left = y0[time_idx]
            p_right = y0[time_idx+1]

            g = geod.Geod(ellps="WGS84")

            k = (time-time_left)/(time_right - time_left)
            forward, back, dist = g.inv(p_left[1], p_left[0], p_right[1], p_right[0])
            p_lon, p_lat, _, = g.fwd(p_left[1], p_left[0], forward, dist*k)
            alt = p_left[2] + k*(p_right[2] - p_left[2])

            res[i] = np.array([p_lat, p_lon, alt])

    return res, x1

def median_filtering(time: np.ndarray, coords:np.ndarray, ids:np.ndarray):
    if (time is None) or (len(time) == 0):
        return time, coords, ids

    thr = 0.1
    k = 10
    masks = []
    for i in range(2):
        y = coords[:, i]
        y2 = np.pad(y, k, mode='edge')
        y_filtered = signal.medfilt(y2, 2 * k - 1)
        y_filtered = y_filtered[k:-k]
        mask = np.abs(y - y_filtered) < thr
        masks.append(mask)

    full_mask = np.logical_and(masks[0], masks[1])

    return np.array(time)[full_mask], coords[full_mask], ids[full_mask]