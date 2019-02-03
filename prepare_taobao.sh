mkdir data
mkdir data/taobao_data/
unzip UserBehavior.csv.zip
mv UserBehavior.csv ./data/taobao_data/
python preprocess/taobao_prepare.py
mkdir dnn_save_path
mkdir dnn_best_model
