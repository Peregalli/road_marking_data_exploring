import ast
import cv2 as cv
import numpy as np
import os
import pandas as pd

from tqdm import tqdm
from image_utils import load_labels_from_json, get_labels_from_mask


def change_mask_label(mask: np.ndarray,
                      old_label: str,
                      dataset: pd.DataFrame) -> np.ndarray:
    new_mask = mask.copy()
    old_color = dataset.loc[dataset['label'] == old_label].iloc[0]['color']
    new_color = dataset.loc[dataset['label'] == old_label].iloc[0]['new_label']
    new_mask[np.all(mask == old_color, axis=-1)] = new_color
    return new_mask


def crop_mask(mask: np.ndarray,
              top: float = 0.5,
              bottom: float = 0.85,
              left: float = 0.0,
              right: float = 0.85) -> np.ndarray:
    H, W, _ = mask.shape
    cropped_mask = mask[int(H*top):int(H*bottom), int(W*left):int(W*right)]
    return cropped_mask


def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except (SyntaxError, ValueError):
        return None


def create_folder(folder: str):
    if not os.path.exists(folder):
        print(f'Creating folder {folder}')
        os.makedirs(folder)


def upload_dataframe(labels_name_frame: pd.DataFrame,
                     new_dataset: pd.DataFrame,
                     fn: str,
                     lables_used: list):
    new_lables = list(set(labels_name_frame[labels_name_frame['label'].isin(lables_used)].category))
    for new_label in new_lables:
        new_dataset.loc[new_dataset['fn'] == fn, new_label] = 1
    return new_dataset


def transform_apolloscape_labels(road: int,
                                 records: int,
                                 camara: int,
                                 dst_folder: str = 'lane_marking_dataset_v1'):

    CURRENT_DPATH = os.path.abspath(os.path.dirname("__file__"))
    DATA_ROOT = os.path.abspath(os.path.join(CURRENT_DPATH, os.pardir, 'Data'))

    dataset_v1 = pd.read_csv('apolloscape_labels_v1.csv')
    dataset_v1['color'] = dataset_v1['color'].apply(safe_literal_eval)
    dataset_v1['new_label'] = dataset_v1['new_label'].apply(safe_literal_eval)
    old_labels = load_labels_from_json('dataset_labels.json')['apolloscape_lanemark']['labels']

    create_folder(dst_folder)
    new_dataset = pd.DataFrame(columns=['fn', 'Road', 'Record', 'lane', 'stopping', 'zebra', 'void'])

    for record in range(1, records):
        print(f'Road {road:02d} - Record {record:03d} - Camera {camara:d}')
        folder_to_labels = os.path.join(DATA_ROOT, 'ApoloScape - Lane Segmentation', f'Labels_road{road:02d}/Label/Record{record:03d}/Camera {camara:d}')
        fn_labels = sorted(os.listdir(folder_to_labels))
        new_dataset['fn'] = fn_labels
        new_dataset['Road'] = road
        new_dataset['Record'] = record
        new_dataset[['lane', 'stopping', 'zebra', 'void']] = 0

        for fn in tqdm(fn_labels):
            mask = cv.imread(os.path.join(folder_to_labels, fn))
            mask = crop_mask(mask)
            lables_used = get_labels_from_mask(mask, old_labels)
            new_mask = mask[:, :, ::-1].copy()
            for label_used in lables_used:
                new_mask = change_mask_label(new_mask, label_used, dataset_v1)
            new_dataset = upload_dataframe(dataset_v1, new_dataset, fn, lables_used)
            cv.imwrite(os.path.join(dst_folder, fn), new_mask)
        new_dataset.to_csv(os.path.join(dst_folder, f'labels_road{road:02d}_record{record:03d}_camera{camara:d}.csv'), index=False)
    return


def main():
    road = 4
    records = 24
    camara = 6

    transform_apolloscape_labels(road, records, camara)
