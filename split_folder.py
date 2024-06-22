import argparse
import glob
import logging
import pandas as pd
from  shutil import copy 
from os import path, mkdir
from transform_apolloscape_labels import split_sets
from utils import setup_logging

def copy_csv_files(src_folder, dst_folder, path_csv):
    files = pd.read_csv(path_csv, header = None)
    print(f'Total {path_csv} files: {len(files)}')
    mask_src_folder = src_folder.replace('images','masks')
    subdir = path.basename(path_csv).split('.')[0]
    if not path.exists(dst_folder):
        mkdir(dst_folder)
    
    dst_folder = path.join(dst_folder, subdir)
    if not path.exists(dst_folder):
        mkdir(dst_folder)
    mask_dst_foder = path.join(dst_folder, 'masks')
    img_dst_foder = path.join(dst_folder, 'images')
    if not path.exists(mask_dst_foder) or not path.exists(img_dst_foder):
        mkdir(mask_dst_foder)
        mkdir(img_dst_foder)
    for i, file in files.iterrows():
        file = file[0]
        img_src = path.join(src_folder,file)
        mask_src = path.join(mask_src_folder, file.replace('.jpg','_bin.png'))
        copy(img_src, img_src.replace(src_folder,img_dst_foder))
        copy(mask_src, mask_src.replace(mask_src_folder,mask_dst_foder))



def split_folder(folder_data: str, dataset_name: str):
    src_folder = path.join(folder_data, dataset_name)
    assert path.exists(path.join(src_folder,"train.csv")), f"file train.csv not exist!"
    assert path.exists(path.join(src_folder,"test.csv")), f"file test.csv not exist!"
    assert path.exists(path.join(src_folder,"valid.csv")), f"file valid.csv not exist!"
    src_folder = path.join(src_folder, 'images')
    dst_folder = '/data/datos_imm/lane_segmantation_dataset_v3/'
    copy_csv_files(src_folder,
                   dst_folder,
                   path_csv=path.join(folder_data, dataset_name,"train.csv"))
    copy_csv_files(src_folder,
                   dst_folder,
                   path_csv=path.join(folder_data, dataset_name,"test.csv"))
    copy_csv_files(src_folder,
                   dst_folder,
                   path_csv=path.join(folder_data, dataset_name,"valid.csv"))
    return


def main():
    parser = argparse.ArgumentParser(
        description='Get information of generated data.')

    parser.add_argument('root_to_data',
                        type=str,
                        help='Root directory to ApolloScape dataset')

    args = parser.parse_args()
    split_folder(args.root_to_data, 'lane_marking_dataset_v3')


if __name__ == '__main__':
    main()
