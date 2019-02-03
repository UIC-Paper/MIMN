# Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction
Implementation of Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction using tensorflow

## Prerequisites
- Python 2.x
- Tensorflow 1.4

## Data
- [Amazon Book Data](http://jmcauley.ucsd.edu/data/amazon/)<br/>
- [Taobao Data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1)

##Getting Started
First we need to prepare data.<br/>

### Amazon Prepare
- You can get the raw Amazon data prepared<br/>
```
sh prepare_amazon.sh
```
- Because getting and processing the data is time consuming，we had processed Amazon data and upload it for you.<br/>
```
sh prepare_ready_data.sh
```

### Taobao Prepare
First download [Taobao Data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1) 
to get "UserBehavior.csv.zip", then execute the following command.
```
sh prepare_taobao.sh
```

## Running
```
usage: train_book.py|train_taobao.py  [-h] [-p TRAIN|TEST] [--random_seed RANDOM_SEED]
                     [--model_type MODEL_TYPE] [--memory_size MEMORY_SIZE]
                     [--mem_induction MEM_INDUCTION]
                     [--util_reg UTIL_REG]
```


### Base Model
The example for DNN
```
python script/train_book.py -p train --random_seed 19 --model_type DNN
python script/train_book.py -p test --random_seed 19 --model_type DNN
```
The model below had been supported: 
- DNN 
- PNN 
- DIN
- GRU4REC
- ARNN
- RUM
- DIEN
- DIEN_with_neg

### MIMN
You can train MIMN with different parameter setting:<br/>
- MIMN Basic
```
python script/train_taobao.py -p train --random_seed 19 --model_type MIMN --memory_size 4 --mem_induction 0 --util_reg 0
```

- MIMN with Memory Utilization Regularization
```
python script/train_taobao.py -p train --random_seed 19 --model_type MIMN --memory_size 4 --mem_induction 0 --util_reg 1
```

- MIMN with Memory Utilization Regularization and Memory Induction Unit
```
python script/train_taobao.py -p train --random_seed 19 --model_type MIMN --memory_size 4 --mem_induction 1 --util_reg 1
```

- MIMN with Auxiliary Loss
```
python script/train_taobao.py -p train --random_seed 19 --model_type MIMN_with_neg --memory_size 4 --mem_induction 0 --util_reg 0
```
If you want to train Amazon Data, you just need replace above train_taobao.py to train_book.py


