import argparse
import pandas as pd

from src.models.train_utils import TrainConfig
from sklearn.model_selection import train_test_split 
from src.data.dfs_schemas import ProcessedDataframeSchema


MAX_SEQ_LEN = 128

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Arguments for training')
    parser.add_argument('--csv_path', type = str, required = True)
    parser.add_argument('--val_set_size', type = float, required = True)
    parser.add_argument('--max_epochs', type = int, required = True)
    parser.add_argument('--batch_size', type = int, required = True)
    parser.add_argument('--initial_lr', type = float, required = True)
    parser.add_argument('--log_path', type = str, required = True)
    parser.add_argument('--log_name', type = str, required = True)
    parser.add_argument('--checkpoint', type = bool, required = False, default = False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--checkpoint_path', type = str, required = False)
    parser.add_argument('--load_config', type = bool, required = False, default = False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--config_path', type = str, required = False)
    parser.add_argument('--n_hidden_units', type = int, required = False)
    parser.add_argument('--hidden_units', type = list, required = False)
    parser.add_argument('--use_dropout', type = bool, required = False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--dropout_prob', type = float, required = False)
    parser.add_argument('--bert_pooled_output', type = int, required = False)

    train_config = TrainConfig(parser)

    # LOAD DATA
    processed_schema = ProcessedDataframeSchema()
    df = pd.read_csv(train_config.csv_path, sep = ',')

    # Validate Correct DataFrame 
    validation_warnings = processed_schema.check_schema(df)
    if len(validation_warnings):
        warnings_str = '\n'.join([warning.message for warning in validation_warnings])
        raise RuntimeError(
            'dataframe in csv_path dont match the required schema'
            + ' id you use data_preprocessing script? \n {}'.format(warnings_str)
        )
    
    # Validation split
    df_train, df_val = train_test_split(
        df,
        test_size = train_config.val_set_size, 
        stratify = df.original_label.values,
        random_state = 42 #always use the same validation set
    ) 
    print('Using a training set of {} samples, validating with {} samples'.format(df_train.shape[0], df_val.shape[0]))










