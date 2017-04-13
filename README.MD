训练
```
time python main.py --train_data_path='./kaggle_train/train.csv' --mode=train  --model_param_path='./model/kaggle_train_result' --eval_data_path='./kaggle_train/test.csv'
```
计算
```
;time python main.py --train_data_path='./kaggle_train/train.csv' --mode=eval  --model_param_path='./model/kaggle_train_result' --eval_data_path='./kaggle_train/test.csv' 
```