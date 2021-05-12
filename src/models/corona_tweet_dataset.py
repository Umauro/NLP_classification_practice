import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset

class CoronaTweetsDataset(Dataset):
    """Pytorch Dataset class to load corona tweets datasets"""
    def __init__(self, df, tokenizer, max_seq_len):
        """
            Parameters

            df: Dataframe
                Dataframe containing CoronaTweets dataset.
            tokenizer: BertTokenizer
                class to tokenize the tweets text.
            max_seq_len: int
                max sequence length considerer in BertModel
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
            Tokenize "idx" tweet, and return its ids, mask, token_type_ids and labels tensors
        """
        if torch.is_tensor(idx):
            idx = idx.toList()
        
        text = self.df.OriginalTweet.iloc[idx]
        labels = self.df[['label_negative','label_neutral','label_positive']].iloc[[idx]].values
        tokenized_text = self.tokenizer(
            text,
            max_length = self.max_seq_len,
            padding = 'max_length',
            return_tensors='pt',
            truncation=True
        )

        sample = {
            'input_ids': tokenized_text['input_ids'],
            'attention_mask': tokenized_text['attention_mask'],
            'token_type_ids': tokenized_text['token_type_ids'],
            'labels': torch.tensor(labels, dtype = torch.long)
        }

        return sample 