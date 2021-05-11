"""
    Pre-process dataset dataframes
"""
import pandas as pd
from data_utils import (
    preprocessing_pipeline
)

DF_TRAIN_PATH = '../../data/raw/Corona_NLP_train.csv'
DF_TEST_PATH = '../../data/raw/Corona_NLP_test.csv'

OUTPUT_TRAIN_PATH = '../../data/processed/df_train.csv'
OUTPUT_TEST_PATH = '../../data/processed/df_test.csv'


df_train = pd.read_csv(DF_TRAIN_PATH,sep = ',')
df_test = pd.read_csv(DF_TEST_PATH, sep = ',')

df_train = preprocessing_pipeline(df_train)
print('Train set preprocessed')
df_test = preprocessing_pipeline(df_test)
print('Test set preprocessed')

df_train.to_csv(OUTPUT_TRAIN_PATH, sep = ',', index = False)
df_test.to_csv(OUTPUT_TEST_PATH, sep= ',', index= False)

