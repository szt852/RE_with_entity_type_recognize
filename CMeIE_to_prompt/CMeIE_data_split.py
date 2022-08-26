from random import *
'''
---句子长度列表长度: 54210
---最长句子字符个数： 297 最短句子字符个数： 4
---数据数量： 54210
---train_split_spot: 43368
---dev_split_spot: 48789
---train_list的数量： 43368
---dev_list的数量： 5421
---test_list的数量： 5421
'''
import random
def data_split(CMeIE_ok_path,train_rate,dev_rate,test_rate):
    '''读取CMeIE_ok_path'''
    with open(CMeIE_ok_path, 'r', encoding='utf-8') as f:
        all_list = f.readlines()
    f.close()

    '''查看一下句子的长度'''
    len_list = []
    for sentence in all_list:
        sentence = eval(sentence)
        len_list.append(len(sentence['token']))
    # print("---句子长度列表:",len_list)
    print("---句子长度列表长度:",len(len_list))
    print("---最长句子字符个数：",max(len_list),"最短句子字符个数：",min(len_list))
    # print(all_list)
    print("---数据数量：",len(all_list))
    random.shuffle(all_list)
    # print(all_list)

    '''训练集的切分点'''
    train_split_spot = int((train_rate / (train_rate + dev_rate + test_rate)) * len(all_list))
    dev_split_spot = int(train_split_spot+((len(all_list)-train_split_spot)/2))
    print("---train_split_spot:",train_split_spot)
    print("---dev_split_spot:",dev_split_spot)

    '''切分'''
    train_list = all_list[:train_split_spot]
    dev_list = all_list[train_split_spot:dev_split_spot]
    test_list = all_list[dev_split_spot:]
    print("---train_list的数量：",len(train_list))
    print("---dev_list的数量：",len(dev_list))
    print("---test_list的数量：",len(test_list))
    return train_list,dev_list,test_list



'''以8：1：1对数据进行切分'''
train_rate = 8
dev_rate = 1
test_rate = 1
CMeIE_ok_path = '../datasets/CMeIE/CMeIE_ok.txt'
train_list,dev_list,test_list = data_split(CMeIE_ok_path, train_rate, dev_rate, test_rate)

'''保存数据'''
CMeIE_train_path = '../datasets/CMeIE/train.txt'
with open(CMeIE_train_path,'w+',encoding='utf-8') as f:
    for i in train_list:
        f.write(str(i))
f.close()

CMeIE_dev_path = '../datasets/CMeIE/dev.txt'
with open(CMeIE_dev_path,'w+',encoding='utf-8') as f:
    for i in dev_list:
        f.write(str(i))
f.close()

CMeIE_test_path = '../datasets/CMeIE/test.txt'
with open(CMeIE_test_path,'w+',encoding='utf-8') as f:
    for i in test_list:
        f.write(str(i))
f.close()