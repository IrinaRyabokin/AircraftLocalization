import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import distance
import argparse

def calc_distances

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        default='result/result_filter_bad_measurements=False_interpolate_between_mlat=True_synchronize_sensors=True_mlat_median_filtering=True.csv',
                        required=False)
    args = parser.parse_args()


    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    gt = pd.read_csv('data/round1_ground_truth.csv')
    pred = pd.read_csv(args.filename)

    gt_pred = gt.merge(pred, on='id', how='outer', suffixes=('_gt', '_pred'))

    gt_pred_dist = calc_distances(gt_pred)

    #
    #
    # distances = []
    #
    # for _, row in gt_pred2.iterrows():
    #     gt_pos = (row['latitude_gt'], row['longitude_gt'])
    #     pred_pos = (row['latitude_pred'], row['longitude_pred'])
    #
    #     dist = distance(gt_pos, pred_pos).m
    #     distances.append(dist)
    #
    # gt_pred2['dist'] = distances
    #
    # print(gt_pred2.sort_values(by='dist', ascending=False).head(10))
    #
    # for aircraft, group in gt_pred2.groupby('aircraft'):
    #     g = group.sort_values(by='id')
    #     plt.plot(g['longitude_gt'], g['latitude_gt'], label='gt')
    #     plt.plot(g['longitude_pred'], g['latitude_pred'], label='pred')
    #     title = f"aircraft {aircraft} maxerr {g['dist'].values.max()}"
    #     plt.title(title)
    #     plt.legend()
    #     plt.show()
    #
    #
