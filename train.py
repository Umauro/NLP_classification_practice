import argparse
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.train_utils import TrainConfig
from src.data.dfs_schemas import ProcessedDataframeSchema
from src.models.corona_tweet_dataset import CoronaTweetsDataset
from src.models.bert_classifier_model import BertClassifierConfig, BertClassifierModel


MAX_SEQ_LEN = 64

  
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
    parser.add_argument('--n_lstm', type = int, required = False)
    parser.add_argument('--use_dropout', type = bool, required = False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--dropout_prob', type = float, required = False)
    parser.add_argument('--bert_pooled_output', type = int, required = False)
    parser.add_argument('--freeze_bert', type = bool, required = False, default = False, action = argparse.BooleanOptionalAction)

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
    print(
        'Using a training set of {} samples, validating with {} samples'.format(
            df_train.shape[0], 
            df_val.shape[0]
        )
    )

    # Dataset and DataLoaders
    # TODO: download this and load from local file xD
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    train_dataset = CoronaTweetsDataset(df_train, tokenizer, MAX_SEQ_LEN)
    val_dataset = CoronaTweetsDataset(df_val, tokenizer, MAX_SEQ_LEN)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = train_config.batch_size,
        shuffle = True,
        num_workers = 4
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = train_config.batch_size,
        num_workers = 4
    )

    # Classification model
    if train_config.load_config:
        model_config = BertClassifierConfig.load_from_json(
            train_config.config_path
        )
    else:
        model_config = BertClassifierConfig.load_from_train_config(
            train_config
        )

    model = BertClassifierModel(model_config)

    # Configure Trainer
    callback_list = list()
    if train_config.checkpoint:
        callback_list.append(ModelCheckpoint(
            dirpath = train_config.checkpoint_path,
            filename = '{epoch}-{train_loss}-{val_loss}',
            monitor = 'val_loss',
            save_top_k = 3, #TODO: pass this as argument,
            mode = 'min'
        ))
    
    callback_list.append(
        EarlyStopping(monitor='val_loss')
    )


    trainer = Trainer(
        max_epochs = train_config.max_epochs,
        gpus = -1, #TODO: pass this as argument
        auto_select_gpus = True, #TODO: same,
        callbacks = callback_list,
        logger = TensorBoardLogger(
            save_dir = train_config.log_path,
            name = train_config.log_name
        )
    )
    
    # Train
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save Model Config
    model_config.save_config(
        '{}/{}_config.json'.format(
            train_config.checkpoint_path,
            train_config.log_name
        )
    )