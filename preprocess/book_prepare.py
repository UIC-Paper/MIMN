import random
import numpy as np
import cPickle as pkl


Train_handle = open("./data/book_data/book_train.txt",'w')
Test_handle = open("./data/book_data/book_test.txt",'w')
Feature_handle = open("./data/book_data/book_feature.pkl",'w')

def produce_neg_item_hist_with_cate(train_file, test_file):
    item_dict = {}
    sample_count = 0
    hist_seq = 0
    for line in train_file:
        units = line.strip().split("\t")
        item_hist_list = units[4].split(",")
        cate_hist_list = units[5].split(",")
        hist_list = zip(item_hist_list, cate_hist_list)
        hist_seq = len(hist_list)
        sample_count += 1
        for item in hist_list:
            item_dict.setdefault(str(item),0)
            
    for line in test_file:
        units = line.strip().split("\t")
        item_hist_list = units[4].split(",")
        cate_hist_list = units[5].split(",")
        hist_list = zip(item_hist_list, cate_hist_list)
        hist_seq = len(hist_list)
        sample_count += 1
        for item in hist_list:
            item_dict.setdefault(str(item),0)
            
    #print item_dict.keys()[:10]   
    del(item_dict["('0', '0')"])
    neg_array = np.random.choice(np.array(item_dict.keys()), (sample_count, hist_seq+20))
    neg_list = neg_array.tolist()
    sample_count = 0
    
    for line in train_file:
        units = line.strip().split("\t")
        item_hist_list = units[4].split(",")
        cate_hist_list = units[5].split(",")
        hist_list = zip(item_hist_list, cate_hist_list)
        hist_seq = len(hist_list)
        neg_hist_list = []
        for item in neg_list[sample_count]:
            item = eval(item)
            if item not in hist_list:
                neg_hist_list.append(item)
            if len(neg_hist_list) == hist_seq:
                break
        sample_count += 1
        neg_item_list, neg_cate_list = zip(*neg_hist_list)
        Train_handle.write(line.strip() + "\t" + ",".join(neg_item_list) + "\t" + ",".join(neg_cate_list) + "\n" )
        
    for line in test_file:
        units = line.strip().split("\t")
        item_hist_list = units[4].split(",")
        cate_hist_list = units[5].split(",")
        hist_list = zip(item_hist_list, cate_hist_list)
        hist_seq = len(hist_list)
        neg_hist_list = []
        for item in neg_list[sample_count]:
            item = eval(item)
            if item not in hist_list:
                neg_hist_list.append(item)
            if len(neg_hist_list) == hist_seq:
                break
        sample_count += 1
        neg_item_list, neg_cate_list = zip(*neg_hist_list)
        Test_handle.write(line.strip() + "\t" + ",".join(neg_item_list) + "\t" + ",".join(neg_cate_list) + "\n" )



def generate_sample_list():
	output_f = open("./data/book_data/long_local_train_splitByUser",'w')
	output_f_test = open("./data/book_data/long_local_test_splitByUser",'w')
	
	for line in file("./data/book_data/local_train_splitByUser"):
	    units = line.strip().split("\t")
	    length = len(units[4].split("\002"))
	    if length >= 20:
	        output_f.write(line)
	for line in file("./data/book_data/local_test_splitByUser"):
	    units = line.strip().split("\t")
	    length = len(units[4].split("\002"))
	    if length >= 20:
	        output_f_test.write(line)

	output_f.close()
	output_f_test.close()

	user_dict = {}
	feature_index = 0
	cat_dict = {}
	item_dict = {}
	max_len = 100
	for line in file("./data/book_data/long_local_train_splitByUser"):
	    label, user, item, cate, item_hist, cate_hist = line.strip().split("\t")
	    user_dict.setdefault(user, 0)
	    cat_dict.setdefault(cate, 0)
	    item_dict.setdefault(item, 0)
	    for ii in item_hist.split("\002"):
	        item_dict.setdefault(ii, 0)
	    for cc in cate_hist.split("\002"):
	        cat_dict.setdefault(cc, 0)
	        
	for line in file("./data/book_data/long_local_test_splitByUser"):
	    label, user, item, cate, item_hist, cate_hist = line.strip().split("\t")
	    user_dict.setdefault(user, 0)
	    cat_dict.setdefault(cate, 0)
	    item_dict.setdefault(item, 0)
	    for ii in item_hist.split("\002"):
	        item_dict.setdefault(ii, 0)
	    for cc in cate_hist.split("\002"):
	        cat_dict.setdefault(cc, 0)
	        
	for uu in user_dict.keys():
	    feature_index += 1
	    user_dict[uu] = feature_index
	for ii in item_dict.keys():
	    feature_index += 1
	    item_dict[ii] = feature_index
	for cc in cat_dict.keys():
	    feature_index += 1
	    cat_dict[cc] = feature_index    

	train_sample_list = []
	test_sample_list = []


	feature_total_num = feature_index + 1
	pkl.dump(feature_total_num, Feature_handle)


	for line in file("./data/book_data/long_local_train_splitByUser"):
	    label, user, item, cate, item_hist, cate_hist = line.strip().split("\t")
	    user_code = user_dict[user]
	    item_code = item_dict[item]
	    cate_code = cat_dict[cate]
	    cate_code_hist = []
	    item_code_hist = []
	    for ii in item_hist.split("\002"):
	        item_code_hist.append(item_dict[ii])
	    for cc in cate_hist.split("\002"):
	        cate_code_hist.append(cat_dict[cc]) 
	    if len(item_code_hist) > max_len:
	        item_code_hist = item_code_hist[-max_len:]
	    else:
	        item_code_hist = [0]*(max_len-len(item_code_hist))+item_code_hist
	    if len(cate_code_hist) > max_len:
	        cate_code_hist = cate_code_hist[-max_len:]
	    else:
	        cate_code_hist = [0]*(max_len-len(cate_code_hist))+cate_code_hist
	    train_sample_list.append(str(user_code) + "\t" + str(item_code) + "\t" + str(cate_code) + "\t" + str(label) + "\t" + ",".join(map(str, item_code_hist)) + "\t" +",".join(map(str, cate_code_hist))+"\n")

	for line in file("./data/book_data/long_local_test_splitByUser"):
	    label, user, item, cate, item_hist, cate_hist = line.strip().split("\t")
	    user_code = user_dict[user]
	    item_code = item_dict[item]
	    cate_code = cat_dict[cate]
	    cate_code_hist = []
	    item_code_hist = []
	    for ii in item_hist.split("\002"):
	        item_code_hist.append(item_dict[ii])
	    for cc in cate_hist.split("\002"):
	        cate_code_hist.append(cat_dict[cc]) 
	    if len(item_code_hist) > max_len:
	        item_code_hist = item_code_hist[-max_len:]
	    else:
	        item_code_hist = [0]*(max_len-len(item_code_hist))+item_code_hist
	    if len(cate_code_hist) > max_len:
	        cate_code_hist = cate_code_hist[-max_len:]
	    else:
	        cate_code_hist = [0]*(max_len-len(cate_code_hist))+cate_code_hist
	    test_sample_list.append(str(user_code) + "\t" + str(item_code) + "\t" + str(cate_code) + "\t" + str(label) + "\t" + ",".join(map(str, item_code_hist)) + "\t" +",".join(map(str, cate_code_hist))+"\n")    
    
	train_sample_length_quant = len(train_sample_list)/128*128
	test_sample_length_quant = len(test_sample_list)/128*128
	train_sample_list = train_sample_list[:train_sample_length_quant]
	test_sample_list = test_sample_list[:test_sample_length_quant]
	random.shuffle(train_sample_list)
	return train_sample_list, test_sample_list


if __name__ == "__main__":
	train_sample_list, test_sample_list = generate_sample_list()
	produce_neg_item_hist_with_cate(train_sample_list, test_sample_list)

