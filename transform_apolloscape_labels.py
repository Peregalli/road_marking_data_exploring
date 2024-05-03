import argparse
import ast
import cv2 as cv
import numpy as np
import os
import pandas as pd
import random

from tqdm import tqdm
from image_utils import get_labels_from_mask, preprocessing_blocks
from utils import list_to_csv, load_from_json


def change_mask_label(mask: np.ndarray,
                      old_label: str,
                      dataset: pd.DataFrame) -> np.ndarray:
    new_mask = mask.copy()
    old_color = dataset.loc[dataset['label'] == old_label].iloc[0]['color']
    new_color = dataset.loc[dataset['label'] == old_label].iloc[0]['new_label']
    new_mask[np.all(mask == old_color, axis=-1)] = new_color
    return new_mask


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
                     new_mask: np.ndarray):
    total_pixels = new_mask.shape[0]*new_mask.shape[1]
    colors_used = np.unique(new_mask.reshape((-1, 3)), axis=0)
    new_dataset.loc[len(new_dataset)] = [fn, 0, 0, 0, 0]
    for i, color in enumerate(colors_used):
        cout_pixels = np.sum(np.all(new_mask == color, axis=-1))
        norm_pixels = cout_pixels/total_pixels
        label = labels_name_frame[labels_name_frame['new_label'].apply(lambda x: x == list(color))].iloc[0].category
        new_dataset.loc[new_dataset['fn'] == fn, label] = norm_pixels
    return new_dataset


def init_dataframe(labels_name_frame: pd.DataFrame) -> pd.DataFrame:
    new_dataset = pd.DataFrame(columns=['fn']+list(set(labels_name_frame.category)))
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


def get_records_names_from_road(road: int, root: str) -> list:
    folder = f'ColorImage_road{road:02d}/ColorImage'
    directory = os.path.join(root, folder)
    roads = sorted(os.listdir(directory))
    return roads


def transform_apolloscape_labels(road: int,
                                 records: list,
                                 root_to_data: str,
                                 dst_folder):

    dataset_v1 = pd.read_csv('config/lane_marking_dataset_v1.csv')
    dataset_v1['color'] = dataset_v1['color'].apply(safe_literal_eval)
    dataset_v1['new_label'] = dataset_v1['new_label'].apply(safe_literal_eval)
    old_labels = load_from_json('config/dataset_labels.json')['apolloscape_lanemark']['labels']
    cfg = load_from_json('config/config.json')
    # Create folders
    dst_folder_img = os.path.join(dst_folder, 'images')
    dst_folder_mask = os.path.join(dst_folder, 'masks')
    create_folder(dst_folder_mask)
    create_folder(dst_folder_img)

    for record in records:
        print(f'Road {road:02d} - '+record+f" - Camera {cfg['camera']:d}")
        folder_to_labels = os.path.join(root_to_data, f'Labels_road{road:02d}/Label/{record}/Camera {cfg["camera"]:d}')
        folder_to_images = os.path.join(root_to_data, f'ColorImage_road{road:02d}/ColorImage/{record}/Camera {cfg["camera"]:d}')
        fn_mask = sorted(os.listdir(folder_to_labels))
        new_dataset = init_dataframe(dataset_v1)
        dst_fn = f'labels_road{road:02d}_{record}_camera{cfg["camera"]:d}.csv'

        for fn in tqdm(fn_mask):
            img_fn = fn.replace('_bin.png', '.jpg')
            img = cv.imread(os.path.join(folder_to_images, img_fn))
            mask = cv.imread(os.path.join(folder_to_labels, fn))
            block_imgs, block_masks = preprocessing_blocks(img,
                                                           mask,
                                                           cfg["cell_size"],
                                                           cfg["overlap"])
            # Preprocess data
            for i, block_mask in enumerate(block_masks):
                lables_used = get_labels_from_mask(block_mask, old_labels)
                new_mask = block_mask[:, :, ::-1].copy()
                for label_used in lables_used:
                    new_mask = change_mask_label(new_mask,
                                                 label_used,
                                                 dataset_v1)
                new_fn_mask = fn.replace('_bin.png', f'_{i}_bin.png')
                new_fn_img = img_fn.replace('.jpg', f'_{i}.jpg')
                new_dataset = upload_dataframe(dataset_v1,
                                               new_dataset,
                                               new_fn_mask,
                                               new_mask)

                cv.imwrite(os.path.join(dst_folder_mask, new_fn_mask), new_mask)
                cv.imwrite(os.path.join(dst_folder_img, new_fn_img), block_imgs[i])
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

    roads = [3, 4]
    total_records = [get_records_names_from_road(road, args.root_to_data) for road in roads]
    for i, road in enumerate(roads):
        transform_apolloscape_labels(road,
                                     total_records[i],
                                     args.root_to_data,
                                     args.root_dst)
    split_sets(args.root_dst)


if __name__ == '__main__':
    main()
