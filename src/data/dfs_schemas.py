from pandas_schema import Column, Schema

class ProcessedDataframeSchema:
    """
        Schema defined for Processed dataframes
    """
    def __init__(self):
        self.schema = Schema([
            Column('OriginalTweet'),
            Column('original_label'),
            Column('label_negative'),
            Column('label_neutral'),
            Column('label_positive')
        ])

    def check_schema(self, df):
        return self.schema.validate(df)