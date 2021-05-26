# NLP_classification_practice-
NLP practice for some things: Pytorch/Tensorflow, text classitication, etc

## Train 
~~~
python train.py --csv_path data/processed/df_train.csv --val_set_size 0.05 --max_epochs 4 --batch_size 16 --initial_lr 0.00005 --log_path models/logs --log_name baseline_dropout --checkpoint --checkpoint_path models/baseline_dropout --n_lstm 0 --use_dropout --dropout_prob 0.1 --bert_pooled_output 768
~~~