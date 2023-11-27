import numpy as np
import pandas as pd
from tqdm import tqdm
import json

def get_train_part(df) -> pd.DataFrame:
    return df[np.logical_not(np.isnan(df['latitude']))]

def get_test_part(df) -> pd.DataFrame:
    return df[np.isnan(df['latitude'])]

def unique_aircrafts(df: pd.DataFrame) -> np.ndarray:
    return np.unique(df['aircraft'])

def get_data_for_aircrafts(df, aircrafts):
    return df[df['aircraft'].isin(aircrafts)]

def parse_measurements(all_df: pd.DataFrame, filter_bad_measurements: bool, desc: str) -> pd.DataFrame:
    aircrafts = unique_aircrafts(all_df)

    res = []

    for aircraft_id in tqdm(aircrafts, desc=desc):
        df = get_data_for_aircrafts(all_df, [aircraft_id]).copy(deep=True)

        col_receiver_time = []
        col_receiver_power = []
        col_sensor_ids = []

        for _, row in df.iterrows():
            meas = json.loads(row['measurements'])

            receiver_time = []
            receiver_power = []
            sensor_ids = []

            for m in meas:
                sensor_id, timestamp, power = m
                receiver_time.append(timestamp / 1e9)
                receiver_power.append(power)
                sensor_ids.append(sensor_id)

            receiver_time = np.array(receiver_time)
            receiver_power = np.array(receiver_power)
            sensor_ids = np.array(sensor_ids)

            col_receiver_time.append(receiver_time)
            col_receiver_power.append(receiver_power)
            col_sensor_ids.append(sensor_ids)

        if filter_bad_measurements:
            time_diff_thr = 0.01
            for i in range(len(col_receiver_time)):
                d = col_receiver_time[i].max() - col_receiver_time[i].min()
                if d >= time_diff_thr:
                    filtered_mask = []

                    k = col_receiver_time[i]

                    for j in range(len(col_receiver_time[i])):
                        curr_t = k[j]
                        curr_min = np.delete(k, j).min()

                        filtered_mask.append(abs(curr_min - curr_t) < time_diff_thr)

                    filtered_mask = np.array(filtered_mask, dtype=bool)

                    col_receiver_time[i] = col_receiver_time[i][filtered_mask]
                    col_receiver_power[i] = col_receiver_power[i][filtered_mask]
                    col_sensor_ids[i] = col_sensor_ids[i][filtered_mask]

        measurements = []
        numMeasurements = []
        for i in range(len(col_receiver_time)):
            new_meas = []
            for j in range(len(col_receiver_time[i])):
                new_meas.append([int(col_sensor_ids[i][j]), int(col_receiver_time[i][j]*1e9), int(col_receiver_power[i][j])])
            measurements.append(new_meas)
            numMeasurements.append(len(new_meas))

        df['measurements'] = measurements
        df['numMeasurements'] = numMeasurements
        res.append(df)

    res = pd.concat(res)
    res = res.sort_values(by='id')
    return res