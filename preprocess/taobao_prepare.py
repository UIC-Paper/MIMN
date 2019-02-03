import cPickle as pkl
import pandas as pd
import random
import numpy as np

RAW_DATA_FILE = './data/taobao_data/UserBehavior.csv'
DATASET_PKL = './data/taobao_data/dataset.pkl'
Test_File = "./data/taobao_data/taobao_test.txt"
Train_File = "./data/taobao_data/taobao_train.txt"
Train_handle = open(Train_File, 'w')
Test_handle = open(Test_File, 'w')
Feature_handle = open("./data/taobao_data/taobao_feature.pkl",'w')

MAX_LEN_ITEM = 200

def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    return df

def remap(df):
    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(item_len)))

    df['iid'] = df['iid'].map(lambda x: item_map[x])

    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(item_len, item_len + user_len)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(user_len + item_len, user_len + item_len + cate_len)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(user_len + item_len + cate_len, user_len + item_len + cate_len + btag_len)))
    df['btag'] = df['btag'].map(lambda x: btag_map[x])

    print(item_len, user_len, cate_len, btag_len)
    return df, item_len, user_len + item_len + cate_len + btag_len + 1 #+1 is for unknown target btag


def gen_user_item_group(df, item_cnt, feature_size):
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')

    print("group completed")
    return user_df, item_df


def gen_dataset(user_df, item_df, item_cnt, feature_size, dataset_pkl):
    train_sample_list = []
    test_sample_list = []

    # get each user's last touch point time

    print len(user_df)

    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])
    print("get user last touch time completed")

    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)]

    cnt = 0
    for uid, hist in user_df:
        cnt += 1
        print(cnt)
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        btag_hist = hist['btag'].tolist()
        target_item_time = hist['time'].tolist()[-1]

        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        target_item_btag = feature_size
        label = 1
        test = (target_item_time > split_time)

        # neg sampling
        neg = random.randint(0, 1)
        if neg == 1:
            label = 0
            while target_item == item_hist[-1]:
                target_item = random.randint(0, item_cnt - 1)
                target_item_cate = item_df.get_group(target_item)['cid'].tolist()[0]
                target_item_btag = feature_size


        # the item history part of the sample
        item_part = []
        for i in range(len(item_hist) - 1):
            item_part.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])
        item_part.append([uid, target_item, target_item_cate, target_item_btag])
        # item_part_len = min(len(item_part), MAX_LEN_ITEM)

        # choose the item side information: which user has clicked the target item
        # padding history with 0
        if len(item_part) <= MAX_LEN_ITEM:
            item_part_pad =  [[0] * 4] * (MAX_LEN_ITEM - len(item_part)) + item_part
        else:
            item_part_pad = item_part[len(item_part) - MAX_LEN_ITEM:len(item_part)]
        
        # gen sample
        # sample = (label, item_part_pad, item_part_len, user_part_pad, user_part_len)

        if test:
            # test_set.append(sample)
            cat_list = []
            item_list = []
            # btag_list = []
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
                # cat_list.append(item_part_pad[i][0])
            test_sample_list.append(str(uid) + "\t" + str(target_item) + "\t" + str(target_item_cate) + "\t" + str(label) + "\t" + ",".join(map(str, item_list)) + "\t" +",".join(map(str, cat_list))+"\n")
        else:
            cat_list = []
            item_list = []
            # btag_list = []
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
            train_sample_list.append(str(uid) + "\t" + str(target_item) + "\t" + str(target_item_cate) + "\t" + str(label) + "\t" + ",".join(map(str, item_list)) + "\t" +",".join(map(str, cat_list))+"\n")

    train_sample_length_quant = len(train_sample_list)/256*256
    test_sample_length_quant = len(test_sample_list)/256*256
        
    print "length",len(train_sample_list)
    train_sample_list = train_sample_list[:train_sample_length_quant]
    test_sample_list = test_sample_list[:test_sample_length_quant]
    random.shuffle(train_sample_list)
    print "length",len(train_sample_list)
    return train_sample_list, test_sample_list


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

def main():
    df = to_df(RAW_DATA_FILE)
    df, item_cnt, feature_size = remap(df)
    print "feature_size", item_cnt, feature_size
    feature_total_num = feature_size + 1
    pkl.dump(feature_total_num, Feature_handle)

    user_df, item_df = gen_user_item_group(df, item_cnt, feature_size)
    train_sample_list, test_sample_list = gen_dataset(user_df, item_df, item_cnt, feature_size, DATASET_PKL)
    produce_neg_item_hist_with_cate(train_sample_list, test_sample_list)


if __name__ == '__main__':
    main()
