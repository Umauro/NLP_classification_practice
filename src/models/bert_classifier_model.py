from typing_extensions import runtime
import torch
import torch.nn as nn
import pytorch_lightning as pl
import warnings
import json


class BertClassifierConfig:
    """Class with configuration parameters for BertClassifierModel"""
    def __init__(
        self,
        n_hidden_layers: int = 1,
        hidden_units: list[int] = [512],
        use_dropout: bool = False,
        dropout_prob: float = 0,
        initial_lr: float = 0.001,
        bert_pooled_output: int = 712
    ):
        """
            Parameters

            n_hidden_layers: int
                number of dense hidden layer after Bert Model.
            hidden_units: list of ints
                number of hidden units of each dense hidden layer.
            use_dropout: bool
                usage of dropout in dense layers.
            dropout_prob: float
                dropout probability.
            initial_lr: float
                initial learning rate for optimizer.
            bert_pooled_output: int
                size of Bert pooled output.
        """

        if dropout_prob < 0 or dropout_prob >= 1:
            raise RuntimeError('dropout prob must be between 0 and 1')

        if n_hidden_layers and len(hidden_units) != n_hidden_layers:
            raise RuntimeError('hidden_units len must match with n_hidden_layers')

        if dropout_prob and not use_dropout:
            warnings.warn('dropout_prob is setted but use_dropout is False. Dropout will not be considered.')

        self.n_hidden_layers = n_hidden_layers
        self.hidden_units = hidden_units
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.initial_lr = initial_lr
        self.bert_pooled_output = bert_pooled_output

    def save_config(self, path: str):
        """
            Save actual configuration to json file

            Parameters:

            path: str
                path to save json file
        """
        if path.split('.')[-1] != 'json':
            raise RuntimeError('Path string must end with .json')

        config_dict = {
            "n_hidden_layers": self.n_hidden_layers,
            "hidden_units": self.hidden_units,
            "use_dropout": self.use_dropout,
            "dropout_prob": self.dropout_prob,
            "initial_lr": self.initial_lr,
            "bert_pooled_output": self.bert_pooled_output
        }     
        with open(path, "w") as config_file:
            json.dump(config_dict, config_file)
        print("Config saved correctly in {}".format(path))

    @classmethod
    def load_from_json(cls, path: str):
        """
            Load config from json file

            Parameters:

            path: str
                path to json file with BertClassifier configuration

            Returns an instance of BertClassifierConfig        
        """

        with open(path, "r") as config_file:
            config_dict = json.load(config_file)
        
        config_instance = BertClassifierConfig(
            config_dict['n_hidden_layers'],
            config_dict['hidden_units'],
            config_dict['use_dropout'],
            config_dict['dropout_prob'],
            config_dict['initial_lr'],
            config_dict['bert_pooled_output']
        )

        return config_instance  
