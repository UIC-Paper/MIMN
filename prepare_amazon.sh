mkdir data
mkdir data/book_data/
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
gunzip reviews_Books.json.gz
gunzip meta_Books.json.gz
python preprocess/process_data.py meta_Books.json reviews_Books_5.json
python preprocess/local_aggretor.py
python preprocess/split_by_user.py
python preprocess/generate_voc.py
mv local_train_splitByUser ./data/book_data/ 
mv local_test_splitByUser ./data/book_data/ 
python preprocess/book_prepare.py
mkdir dnn_save_path
mkdir dnn_best_model
