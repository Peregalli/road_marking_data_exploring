import cv2 as cv
import numpy as np
import os

from utils import load_labels_from_json


def overlay_mask(image: np.ndarray,
                 mask: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    # Create a mask with the same dimensions as the image
    overlay = image.copy()

    # Apply the mask with transparency
    cv.addWeighted(mask, alpha, overlay, 1 - alpha, 0, overlay)

    return overlay


def sort_key(filename: str) -> int:
    return int(filename.split('_')[1])


def get_image_and_mask(folder_to_images: str,
                       folder_to_labels: str,
                       fname: str = None,
                       apollo_dataset: bool = False) -> tuple:
    if fname is None:
        imgs_in_folder = os.listdir(folder_to_images)
        i = np.random.randint(0, len(imgs_in_folder))
        fname = imgs_in_folder[i]
    img = cv.imread(f'{folder_to_images}/{fname}')
    mask_name = (fname).replace('.jpg', '.png')
    if apollo_dataset:
        mask_name = (fname).replace('.jpg', '_bin.png')
    mask = cv.imread(f'{folder_to_labels}/{mask_name}')
    return img, mask


def get_labels_from_mask(mask: np.ndarray, dataset_labels: dict):
    labels = []
    for key, value in dataset_labels.items():
        color_to_check = np.array(value)[::-1]
        if np.any(np.all(mask.reshape(-1, 3) == color_to_check, axis=1)):
            labels.append(key)
    return labels


def check_label_is_in_mask(mask: np.ndarray, dataset_labels: dict, label: str):
    color_to_check = np.array(dataset_labels[label])[::-1]
    return np.any(np.all(mask.reshape(-1, 3) == color_to_check, axis=1))


def get_fname_with_label(folder_to_images: str,
                         folder_to_labels: str,
                         dataset_name: str,
                         label: str) -> str:
    imgs_in_folder = os.listdir(folder_to_images)
    dataset_labels = load_labels_from_json('dataset_labels.json')[dataset_name]['labels']
    fnames_with_label = []
    for fname in imgs_in_folder:
        mask_name = (fname).replace('.jpg', '_bin.png')
        mask = cv.imread(f'{folder_to_labels}/{mask_name}')
        if check_label_is_in_mask(mask, dataset_labels, label):
            print(fname)
            fnames_with_label.append(fname)
    return fnames_with_label


def write_masks_labels(mask: np.ndarray,
                       dataset_name: str,
                       folder_to_labels: str = 'dataset_labels.json') -> np.ndarray:

    assert os.path.exists(folder_to_labels), f'{folder_to_labels} does not exist'
    assert dataset_name in load_labels_from_json(folder_to_labels).keys(), f'{dataset_name} not in {folder_to_labels}'

    # Get labels from that dataset
    dataset_labels = load_labels_from_json(folder_to_labels)[dataset_name]['labels']
    lables_used = get_labels_from_mask(mask, dataset_labels)

    mask_with_labels = mask.copy()
    for i, label in enumerate(lables_used):
        H, W, _ = mask.shape
        offset = 60
        h = offset + 60*i
        color = dataset_labels[label][::-1]
        mask_with_labels = cv.putText(mask_with_labels, label,
                                      (50, h), cv.FONT_HERSHEY_SIMPLEX, 2,
                                      color, 3, cv.LINE_AA)
    return mask_with_labels
