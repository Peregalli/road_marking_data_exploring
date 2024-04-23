import argparse
import ast
import cv2 as cv
import numpy as np
import os
import pandas as pd
import random

from tqdm import tqdm
from image_utils import load_labels_from_json, get_labels_from_mask
from utils import list_to_csv


def change_mask_label(mask: np.ndarray,
                      old_label: str,
                      dataset: pd.DataFrame) -> np.ndarray:
    new_mask = mask.copy()
    old_color = dataset.loc[dataset['label'] == old_label].iloc[0]['color']
    new_color = dataset.loc[dataset['label'] == old_label].iloc[0]['new_label']
    new_mask[np.all(mask == old_color, axis=-1)] = new_color
    return new_mask


def crop_mask(mask: np.ndarray,
              x: int = 50,
              y: int = 1250,
              w: int = 1920,
              h: int = 1080) -> np.ndarray:
    H, W, _ = mask.shape
    cropped_mask = mask[y:y+h, x:x+w]
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
                     lables_used: list) -> pd.DataFrame:
    category_used = labels_name_frame[labels_name_frame['label'].isin(lables_used)].category
    new_lables = list(set(category_used))
    for new_label in new_lables:
        new_dataset.loc[new_dataset['fn'] == fn, new_label] = 1
    return new_dataset


def init_dataframe(labels_name_frame: pd.DataFrame,
                   filenames: list) -> pd.DataFrame:
    cols = ['fn'] + list(set(labels_name_frame.category))
    new_dataset = pd.DataFrame(columns=cols)
    new_dataset['fn'] = filenames
    new_dataset[['lane', 'stopping', 'zebra', 'void']] = 0
    return new_dataset


def split_sets(root_to_dataset: str,
               train_perc: float = 0.7,
               valid_perc: float = 0.2):
    root_to_images = os.path.join(root_to_dataset, 'images')
    img_fn = os.listdir(root_to_images)
    random.shuffle(img_fn)
    total = len(img_fn)

    train_end = int(total*train_perc)
    valid_end = int(total*(valid_perc+train_perc))
    img_fn_train = img_fn[:train_end]
    img_fn_valid = img_fn[train_end:valid_end]
    img_fn_test = img_fn[valid_end:]

    list_to_csv(os.path.join(root_to_dataset, 'train.csv'), img_fn_train)
    list_to_csv(os.path.join(root_to_dataset, 'valid.csv'), img_fn_valid)
    list_to_csv(os.path.join(root_to_dataset, 'test.csv'), img_fn_test)

    return


def transform_apolloscape_labels(road: int,
                                 records: int,
                                 camara: int,
                                 root_to_data: str,
                                 dst_folder):

    dataset_v1 = pd.read_csv('apolloscape_labels_v1.csv')
    dataset_v1['color'] = dataset_v1['color'].apply(safe_literal_eval)
    dataset_v1['new_label'] = dataset_v1['new_label'].apply(safe_literal_eval)
    old_labels = load_labels_from_json('dataset_labels.json')['apolloscape_lanemark']['labels']
    # Create folders
    dst_folder_img = os.path.join(dst_folder, 'images')
    dst_folder_mask = os.path.join(dst_folder, 'masks')
    create_folder(dst_folder_mask)
    create_folder(dst_folder_img)

    for record in range(1, records):
        print(f'Road {road:02d} - Record {record:03d} - Camera {camara:d}')
        folder_to_labels = os.path.join(root_to_data, f'Labels_road{road:02d}/Label/Record{record:03d}/Camera {camara:d}')
        folder_to_images = os.path.join(root_to_data, f'ColorImage_road{road:02d}/ColorImage/Record{record:03d}/Camera {camara:d}')
        fn_mask = sorted(os.listdir(folder_to_labels))
        new_dataset = init_dataframe(dataset_v1, fn_mask)
        dst_fn = f'labels_road{road:02d}_record{record:03d}_camera{camara:d}.csv'

        for fn in tqdm(fn_mask):
            img_fn = fn.replace('_bin.png', '.jpg')
            img = cv.imread(os.path.join(folder_to_images, img_fn))
            img = crop_mask(img)
            mask = cv.imread(os.path.join(folder_to_labels, fn))
            mask = crop_mask(mask)
            lables_used = get_labels_from_mask(mask, old_labels)
            new_mask = mask[:, :, ::-1].copy()
            for label_used in lables_used:
                new_mask = change_mask_label(new_mask, label_used, dataset_v1)
            new_dataset = upload_dataframe(dataset_v1, new_dataset, fn, lables_used)

            cv.imwrite(os.path.join(dst_folder_mask, fn), new_mask)
            cv.imwrite(os.path.join(dst_folder_img, img_fn), img)
        new_dataset.to_csv(os.path.join(dst_folder, dst_fn), index=False)
    return


def main():
    parser = argparse.ArgumentParser(description='Preprocess ApolloScape dataset.')

    parser.add_argument('root_to_data',
                        type=str,
                        help='Root directory to ApolloScape dataset')
    parser.add_argument('root_dst',
                        type=str,
                        help='Directory destination for new data')

    args = parser.parse_args()

    roads = [2, 3]
    total_records = [49, 58]
    camara = 6
    for i, road in enumerate(roads):
        transform_apolloscape_labels(road,
                                     total_records[i],
                                     camara,
                                     args.root_to_data,
                                     args.root_dst)
    split_sets(args.root_dst)


if __name__ == '__main__':
    main()
