import argparse
import glob
import pandas as pd

from os import path
from transform_apolloscape_labels import split_sets


def get_info_from_generated_data(folder_data: str):
    apolloscape_labels = pd.read_csv('config/apolloscape_labels_v1.csv')
    pattern = path.join(folder_data, 'labels*.csv')
    labels_files = glob.glob(pattern)
    df = pd.DataFrame(columns=list(set(apolloscape_labels.category)))
    void_images = 0
    total_images = 0
    for file_names in labels_files:
        df_temp = pd.read_csv(file_names, index_col=0)
        df.loc[len(df)] = df_temp.mean()
        void_images += sum(df_temp.void == 1)
        total_images += len(df_temp)
    dataset_prop = df.mean().values
    weights = (1/len(df.columns))/dataset_prop
    norm_weights = weights/sum(weights)
    print('Mean pixel appearence :\n', df.mean())
    formatted_numbers = ['{:.5f}'.format(number) for number in norm_weights]
    print(f'NORMALIZED WEIGHTS : {formatted_numbers}')
    print('Total images with void mask : ', void_images)
    print('Total images : ', total_images)
    print(f'Percentage of void images : {int(void_images/total_images*100)}%')
    split_sets(folder_data)
    return


def main():
    parser = argparse.ArgumentParser(
        description='Get information of generated data.')

    parser.add_argument('root_to_data',
                        type=str,
                        help='Root directory to ApolloScape dataset')

    args = parser.parse_args()

    get_info_from_generated_data(args.root_to_data)


if __name__ == '__main__':
    main()
