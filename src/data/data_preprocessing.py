"""
    Pre-process dataset dataframes
"""
import pandas as pd
import argparse
from .data_utils import (
    preprocessing_pipeline
)

parser = argparse.ArgumentParser(
    description='Arguments for data pre-processing'
)

parser.add_argument(
    '--csv_path',
    type=str,
    required=True,
    help='path for csv file with training dataset'
)

parser.add_argument(
    '--output_path',
    type=str,
    required=True,
    help='path for preprocessed dataset'
)

args = parser.parse_args()

CSV_PATH = args.csv_path
OUTPUT_PATH = args.output_path

try:
    df = pd.read_csv(CSV_PATH,sep = ',')
    df = preprocessing_pipeline(df)
    print('Dataset preprocessed')

    try:
        df.to_csv(OUTPUT_PATH, sep = ',', index = False)
        print('Dataset saved to {}'.format(OUTPUT_PATH))
    except FileNotFoundError as error:
        print('Error trying to save the csv pre-processed dataset')
        print(error)

except (FileNotFoundError, IOError) as error:
    print('Error trying to read csv file')
    print(error)
    


