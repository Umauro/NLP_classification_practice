import warnings

def setting_argument_warning(bool_arg: str, arg_name: str, arg_default: str):
    warnings.warn(
        '{} is disabled but {} was not specified. Setting n_hidden_units to {}'
            .format(bool_arg, arg_name, arg_default)
    )

class TrainConfig:
    """
        Class with training configuration params
    """
    def __init__(self, parser):
        args = parser.parse_args()
        self.parser = parser
        self.csv_path = args.csv_path
        self.val_set_size = args.val_set_size
        self.max_epochs = args.max_epochs
        self.batch_size = args.batch_size
        self.initial_lr = args.initial_lr
        self.log_path = args.log_path
        self.log_name = args.log_name
        self.checkpoint = args.checkpoint
        self.checkpoint_path = args.checkpoint_path
        self.load_config = args.load_config
        self.config_path = args.config_path
        self.n_lstm = args.n_lstm
        self.use_dropout = args.use_dropout
        self.dropout_prob = args.dropout_prob
        self.bert_pooled_output = args.bert_pooled_output
        self.freeze_bert = args.freeze_bert
        self.validate_config()

    def validate_config(self):
        """
            Validate some config params values
        """
        if self.val_set_size <= 0:
            self.parser.error('val_set_size must be greater than 0')

        if self.max_epochs < 1:
            warnings.warn('max_epochs must be greater than 0. Setting max_epochs to 1.')
            self.max_epochs = 1

        if self.batch_size < 1:
            warnings.warn('batch_size must be greater than 0. Setting batch_size to 32.')
            self.batch_size = 32

        if self.initial_lr < 0:
            self.parser.error('initial lr must be positive.')

        if self.checkpoint and self.checkpoint_path is None:
            self.parser.error('--checkpoint is activated but checkpoint_path was not specified.')

        if self.load_config and self.config_path is None:
            self.parser.error('--load_config is activated but config_path was not specified.')

        if not self.load_config:
            if self.n_lstm is None:
                if self.freeze_bert:
                    setting_argument_warning('load_config', 'n_lstm', '2')
                    self.n_lstm = 2
                else:
                    setting_argument_warning('load_config','n_lstm',0)
            
            if self.use_dropout is None:
                setting_argument_warning('load_config', 'use_dropout', 'False')
                self.use_dropout = False
            
            if self.dropout_prob is None:
                setting_argument_warning('load_config', 'dropout_prob', '0')
                self.dropout_prob = 0
            
            if self.bert_pooled_output is None:
                setting_argument_warning('load_config', 'bert_pooled_output', '768')
                self.bert_pooled_output = 768
    