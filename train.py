import argparse
import warnings
import pandas as pd

from sklearn.model_selection import train_test_split 
from src.data.dfs_schemas import ProcessedDataframeSchema


MAX_SEQ_LEN = 128

def setting_argument_warning(bool_arg: str, arg_name: str, arg_default: str):
    warnings.warn(
        '{} is disabled but {} was not specified. Setting n_hidden_units to {}'
            .format(bool_arg, arg_name, arg_default)
    )
    
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

    args = parser.parse_args()
    csv_path = args.csv_path
    val_set_size = args.val_set_size
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    log_path = args.log_path
    log_name = args.log_name
    checkpoint = args.checkpoint
    checkpoint_path = args.checkpoint_path
    load_config = args.load_config
    config_path = args.config_path
    n_hidden_units = args.n_hidden_units
    hidden_units = args.hidden_units
    use_dropout = args.use_dropout
    dropout_prob = args.dropout_prob
    bert_pooled_output = args.bert_pooled_output

    if val_set_size <= 0:
        parser.error('val_set_size must be greater than 0')

    if max_epochs < 1:
        warnings.warn('max_epochs must be greater than 0. Setting max_epochs to 1.')
        max_epochs = 1

    if batch_size < 1:
        warnings.warn('batch_size must be greater than 0. Setting batch_size to 32.')
        batch_size = 32

    if initial_lr < 0:
        parser.error(message = 'initial lr must be positive.')

    if checkpoint and checkpoint_path is None:
        parser.error(message = '--checkpoint is activated but checkpoint_path was not specified.')

    if load_config and config_path is None:
        parser.error(message = '--load_config is activated but config_path was not specified.')

    if not load_config:
        if n_hidden_units is None:
            setting_argument_warning('load_config', 'n_hidden_units', '1')
            n_hidden_units = 1
        
        if hidden_units is None:
            setting_argument_warning('load_config', 'hidden_units', '[512]')
            hidden_units = [512]
        
        if use_dropout is None:
            setting_argument_warning('load_config', 'use_dropout', 'False')
            use_dropout = False
        
        if dropout_prob is None:
            setting_argument_warning('load_config', 'dropout_prob', '0')
            dropout_prob = 0
        
        if bert_pooled_output is None:
            setting_argument_warning('load_config', 'bert_pooled_output', '768')
            bert_pooled_output = 768

    # LOAD DATA
    processed_schema = ProcessedDataframeSchema()
    df = pd.read_csv(csv_path, sep = ',')

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
        test_size = val_set_size, 
        stratify = df.original_label.values,
        random_state = 42 #always use the same validation set
    ) 
    print('Using a training set of {} samples, validating with {} samples'.format(df_train.shape[0], df_val.shape[0]))










