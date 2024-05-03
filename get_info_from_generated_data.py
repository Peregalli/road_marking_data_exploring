import argparse
import glob
import logging
import pandas as pd

from os import path
from transform_apolloscape_labels import split_sets
from utils import setup_logging


def get_info_from_generated_data(folder_data: str, dataset_name: str):
    apolloscape_labels = pd.read_csv(f'config/{dataset_name}.csv')
    pattern = path.join(folder_data, 'labels*.csv')
    labels_files = glob.glob(pattern)
    df = pd.DataFrame(columns=list(set(apolloscape_labels.category)))
    void_images = 0
    total_images = 0
    # Call the setup_logging function
    setup_logging(dataset_name)
    logging.info('DATA VERSION : ' + dataset_name)
    print(apolloscape_labels)
    print('FOLDERS :')
    for file_names in labels_files:
        print(file_names)
        df_temp = pd.read_csv(file_names, index_col=0)
        select_tiles = df_temp.index.str.endswith(('_4_bin.png',
                                                   '_5_bin.png',
                                                   '_6_bin.png',
                                                   '_7_bin.png'))
        df_temp = df_temp[select_tiles]
        df.loc[len(df)] = df_temp.mean()
        void_images += sum(df_temp.void == 1)
        total_images += len(df_temp)
    dataset_prop = df.mean().values
    weights = (1/len(df.columns))/dataset_prop
    norm_weights = weights/sum(weights)
    print('MEAN PIXEL APPEARANCE :\n',
          df.mean())
    formatted_numbers = ['{:.5f}'.format(number) for number in norm_weights]
    print(f'NORMALIZED WEIGHTS : {formatted_numbers}')
    print('TOTAL IMAGES : ', total_images)
    print('TOTAL IMAGES WITH TOTAL BACKGROUND : ', void_images)
    print(f'PERCENTAGE OF BACKGROUND IMAGES : {int(void_images/total_images*100)}%')
    split_sets(folder_data)
    logging.shutdown()
    return


def main():
    parser = argparse.ArgumentParser(
        description='Get information of generated data.')

    parser.add_argument('root_to_data',
                        type=str,
                        help='Root directory to ApolloScape dataset')

    args = parser.parse_args()
    get_info_from_generated_data(args.root_to_data, 'lane_marking_dataset_v2')


if __name__ == '__main__':
    main()
