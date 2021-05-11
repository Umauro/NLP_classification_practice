"""
    Utils data function to pre-process dataframes
"""
import pandas as pd
import html 

UNUSED_COLUMNS = ['UserName', 'ScreenName', 'Location', 'TweetAt']

LABELS_DICT = {
    'Extremely Negative':'negative',
    'Negative':'negative',
    'Extremely Positive':'positive',
    'Positive':'positive',
    'Neutral':'neutral'
}

HASHTAG_REGEX = r'\B#([a-z]*[A-Z]*[0-9]*[0-9]*_*)*'
MENTIONS_REGEX = r'\B@([a-z]*[A-Z]*[0-9]*[0-9]*_*)*'
URLS_REGEX = r'\b(http(s)?){1}:\/\/.*'

def aggrupate_labels(df, labels_dict):
    """
        Aggrupate Sentiments labels in negative, neutral, positive
    """
    df['Sentiment'] = df.Sentiment.replace(labels_dict)
    return df

def one_hot_encoding(df, prefix, label_column):
    """
        One Hot encodes Sentiment labels
    """ 
    return pd.get_dummies(df, prefix = prefix, columns = [label_column])

def replace_pattern(df, regex_pattern, replace_value):
    """
        Replace patterns in OriginalTweet column based con regex pattern
    """
    df['OriginalTweet'] = df['OriginalTweet'].str.replace(regex_pattern, replace_value, regex = True)
    return df

def replace_html_characters(df):
    """
        Replace html special characters in OriginalTweet column
    """
    df['OriginalTweet'] = df['OriginalTweet'].apply(html.unescape)
    return df

def drop_columns(df, unused_columns):
    """
        drop unused columns
    """
    return df.drop(columns = unused_columns)

def preprocessing_pipeline(df):
    """
        Apply preprocessing functions
    """
    df_preprocessed = (df.pipe(aggrupate_labels, labels_dict = LABELS_DICT)
                        .pipe(one_hot_encoding, prefix = 'label', label_column = 'Sentiment')
                        .pipe(replace_pattern, regex_pattern = HASHTAG_REGEX, replace_value = 'tag')
                        .pipe(replace_pattern, regex_pattern = MENTIONS_REGEX, replace_value = 'mention')
                        .pipe(replace_pattern, regex_pattern = URLS_REGEX, replace_value = 'url')
                        .pipe(replace_html_characters)
                        .pipe(drop_columns, unused_columns = UNUSED_COLUMNS))
    return df_preprocessed




    

