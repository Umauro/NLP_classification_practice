import warnings
import json

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.modules import dropout
import torchmetrics
from transformers import BertModel
from .train_utils import TrainConfig

class BertClassifierConfig:
    """Class with configuration parameters for BertClassifierModel"""
    def __init__(
        self,
        n_lstm: int = 2,
        use_dropout: bool = False,
        dropout_prob: float = 0,
        initial_lr: float = 0.001,
        bert_pooled_output: int = 768,
        freeze_bert: bool = True
    ):
        """
            Parameters

            n_lstm: int
                number of bidirectional LSTM after Bert Model. Used only when freeze_bert == False
            use_dropout: bool
                usage of dropout in dense layers.
            dropout_prob: float
                dropout probability.
            initial_lr: float
                initial learning rate for optimizer.
            bert_pooled_output: int
                size of Bert pooled output.
            freeze_bert: bool
                Freeze bert parameters during training
        """

        if dropout_prob < 0 or dropout_prob >= 1:
            raise RuntimeError('dropout prob must be between 0 and 1')

        if n_lstm < 0:
            raise RuntimeError('n_lstm must be greater than 0')

        if dropout_prob and not use_dropout:
            warnings.warn('dropout_prob is setted but use_dropout is False. Dropout will not be considered.')

        self.n_lstm = n_lstm
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.initial_lr = initial_lr
        self.bert_pooled_output = bert_pooled_output
        self.freeze_bert = freeze_bert

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
            "n_lstm": self.n_lstm,
            "use_dropout": self.use_dropout,
            "dropout_prob": self.dropout_prob,
            "initial_lr": self.initial_lr,
            "bert_pooled_output": self.bert_pooled_output,
            "freeze_bert": self.freeze_bert
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
            config_dict['n_lstm'],
            config_dict['use_dropout'],
            config_dict['dropout_prob'],
            config_dict['initial_lr'],
            config_dict['bert_pooled_output'],
            config_dict['freeze_bert']
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
            train_config.n_lstm,
            train_config.use_dropout,
            train_config.dropout_prob,
            train_config.initial_lr,
            train_config.bert_pooled_output,
            train_config.freeze_bert
        )

        return config_instance

class ClassificationHead(nn.Module):
    """
        Classification head that receives BertModel output

        If bert parameters are freezed (Bert as feature Extrator),
        uses BiLSTM cells. If we use fine-tunning, it only has a linear layer
    """
    def __init__(self, config: BertClassifierConfig):
        """
            Parameters:

            config: BertClassifierConfig
                object with model configuration parameters
        """
        super().__init__()
        self.linear_in_features = config.bert_pooled_output
        self.lstm = None
        self.dropout = None

        if config.freeze_bert:
            self.lstm = nn.LSTM(
                    input_size = config.bert_pooled_output,
                    hidden_size = config.bert_pooled_output,
                    num_layers = config.n_lstm,
                    dropout = config.dropout_prob,
                    batch_first = True,
                    bidirectional = True
                )
            self.linear_in_features *= 2
        
        if config.use_dropout:
            self.dropout = nn.Dropout(config.dropout_prob)
        
        self.linear = nn.Linear(self.linear_in_features, 3)
        self.linear.apply(self.weight_init)

    def weight_init(self,module):
        """
            Weight initializer for linear layers
        """
        if isinstance(module, nn.Linear):
            # official implementation uses weight from a normal dist with mean 0 and std 0.02
            nn.init.normal_(module.weight.data, 0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
    
    def forward(self,input_tensor):
        """
            Compute Classification head forward pass
        """
        output = input_tensor
        if self.lstm is not None:
            output, _ = self.lstm(output) 
            if self.dropout is not None:
                output = self.dropout(output[:,-1,:]) #only last seq item
        else:
            if self.dropout is not None:
                output = self.dropout(output)
        output = self.linear(output)
        return output

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
        self.lr = config.initial_lr
        self.bert = BertModel.from_pretrained('bert-base-uncased') #TODO: download bert model c:
        self.cls = ClassificationHead(config)
        self.f1_score = torchmetrics.F1(num_classes = 3)
        self.accuracy = torchmetrics.Accuracy(num_classes = 3)
        self.freeze_bert = config.freeze_bert

        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    


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

        if self.freeze_bert:
            selected_bert_output = bert_output[0] # if we use LSTM we need a output with shape batch, seq, feature
        else:
            selected_bert_output = bert_output[1] # else, we use pooled output
        
        output = self.cls(selected_bert_output)
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
        
        if self.freeze_bert:
            selected_bert_output = bert_output[0] 
        else:
            selected_bert_output = bert_output[1] 
        
        output = self.cls(selected_bert_output)
        loss = nn.functional.cross_entropy(output, labels)
        self.log('train_loss', loss)
        
        # Calcule step acc and F1
        softmaxed_output = nn.functional.softmax(output, dim = 1)
        self.accuracy(softmaxed_output, labels)
        self.f1_score(softmaxed_output, labels)
        self.log('train_acc', self.accuracy, on_step = True, on_epoch = False)
        self.log('train_f1', self.f1_score, on_step = True, on_epoch = False)
        
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
            token_type_ids = token_type_ids
        )
        
        if self.freeze_bert:
            selected_bert_output = bert_output[0] # if we use LSTM we need a output with shape batch, seq, feature
        else:
            selected_bert_output = bert_output[1] # else, we use pooled output

        output = self.cls(selected_bert_output)
        loss = nn.functional.cross_entropy(output, labels)
        self.log('val_loss', loss)

        # calcule validation acc and f1
        softmaxed_output = nn.functional.softmax(output, dim = 1)
        self.accuracy(softmaxed_output, labels)
        self.f1_score(softmaxed_output, labels)
        self.log('val_acc', self.accuracy, on_step = True, on_epoch = True)
        self.log('val_f1', self.f1_score, on_step = True, on_epoch = True)



        


    






