import warnings
import json

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from transformers import BertModel
from .train_utils import TrainConfig

class BertClassifierConfig:
    """Class with configuration parameters for BertClassifierModel"""
    def __init__(
        self,
        n_hidden_layers: int = 1,
        hidden_units: list = [512,],
        use_dropout: bool = False,
        dropout_prob: float = 0,
        initial_lr: float = 0.001,
        bert_pooled_output: int = 768
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

    @classmethod
    def load_from_train_config(cls, train_config: TrainConfig):
        """
            Load config from training configuration object

            Parameters:

            train_config: TrainConfig object
                
        """
        config_instance = BertClassifierConfig(
            train_config.n_hidden_layers,
            train_config.hidden_units,
            train_config.use_dropout,
            train_config.dropout_prob,
            train_config.initial_lr,
            train_config.bert_pooled_output
        )

        return config_instance

class BertClassifierModel(pl.LightningModule):
    """
        Class with Bert Classifier Model
    """
    def __init__(self, config: BertClassifierConfig):
        """
            Parameters:

            config: BertClassifierConfig
                object with model configuration parameters
        """
        super().__init__()
        modules = list()
        
        for index in range(config.n_hidden_layers):
            if index != 0:
                in_features = config.hidden_units[index - 1]
            else:
                in_features = config.bert_pooled_output 
                
            out_features = config.hidden_units[index]
            
            modules.append(
                nn.Linear(in_features, out_features)
            )
            modules.append(
                nn.ReLU(inplace = True)
            )
            
            if config.use_dropout:
                modules.append(
                    nn.Dropout(config.dropout_prob)
                )
        modules.append(nn.Linear(config.hidden_units[-1],3))
            
        self.lr = config.initial_lr
        self.bert = BertModel.from_pretrained('bert-base-uncased') #TODO: download bert model c:
        self.cls = nn.Sequential(*modules)
        self.cls.apply(self.weight_init)
        self.f1_score = torchmetrics.F1(num_classes = 3)
        self.accuracy = torchmetrics.Accuracy(num_classes = 3, average = None)
    
    def weight_init(self,module):
        """
            Weight initializer for classification layers
        """
        if isinstance(module, nn.Linear):
            # official implementation uses weight from a normal dist with mean 0 and std 0.02
            nn.init.normal_(module.weight.data, 0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
            Compute the model forward pass

            Parameters:
            
            input_ids: Pytorch Tensor
                input_ids tensor from BertTokenizer
            attention_mask: Pytorch Tensor
                attention_mask tensor from BertTokenizer
            token_type_ids: Pytorch Tensor
                token_type_ids tensor from BertTokenizer
        """
        bert_output = self.bert(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids
        )
        pooled_output = bert_output[1]
        output = self.cls(pooled_output)
        output = nn.functional.softmax(output, dim = 1)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
            Compute the loss training step

            Parameters:

            train_batch: dict
                Dictionary obtained from CoronaTweetsDataset with training samples
        """
        input_ids = train_batch['input_ids']
        attention_mask = train_batch['attention_mask']
        token_type_ids = train_batch['token_type_ids']
        labels = train_batch['labels']
        
        # this is similar to forward, but Pytorch Lightning recommend separate inference from training
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = bert_output[1]
        output = self.cls(pooled_output)
        loss = nn.functional.cross_entropy(output, labels)
        self.log('train_loss', loss)
        
        # Calcule step acc and F1
        softmaxed_output = nn.functional.softmax(output, dim = 1)
        self.accuracy(softmaxed_output, labels)
        self.f1_score(softmaxed_output, labels)
        self.log('train_acc', self.accuracy, on_step = True, on_epoch = False)
        self.log('train_f1',  on_step = True, on_epoch = False)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
            Compute a validation step

            Parameters:

            val_batch: dict
                Dictionary obtained from CoronaTweetsDataset with validation samples
        """
        input_ids = val_batch['input_ids']
        attention_mask = val_batch['attention_mask']
        token_type_ids = val_batch['token_type_ids']
        labels = val_batch['labels']
        
        # this is similar to forward, but Pytorch Lightning recommend separate inference from training
        bert_output = self.bert(
            input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids)
        pooled_output = bert_output[1]
        loss = nn.functional.cross_entropy(self.cls(pooled_output), labels)
        self.log('val_loss', loss)

        # calcule validation acc and f1
        softmaxed_output = nn.functional.softmax(self.cls(pooled_output), dim = 1)
        self.accuracy(softmaxed_output, labels)
        self.f1_score(softmaxed_output, labels)
        self.log('val_acc', on_step = True, on_epoch = True)
        self.log('val_f1', on_step = True, on_epoch = True)



        


    






