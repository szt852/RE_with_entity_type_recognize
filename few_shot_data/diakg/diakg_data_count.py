import json
import random
import os

def data_n_split(origin_train_path,output_train_path,n_split=32):
    # origin_train_data = []
    '''读取文件 diakg_have_re.txt'''
    with open(origin_train_path,'r',encoding='utf-8') as f:
        origin_train_data = f.readlines()
    # print(diakg_have_re)
    f.close()

    re_count=[]
    sentence_list=[]
    for s in origin_train_data:
        # print(type(s))
        global false,true
        false = False
        true = True
        sentence = eval(s)
        # print(type(sentence))
        for i in sentence['spo_list']:
            new_sentence = {}
            new_sentence['text'] = sentence['text']
            tmp_list = []
            tmp_list.append(i)
            new_sentence['spo_list'] = tmp_list
            new_sentence['relation'] = str(new_sentence['spo_list'][0]['subject_type'])+":"+str(new_sentence['spo_list'][0]['object_type']['@value'])
            re_count.append(str(new_sentence['spo_list'][0]['subject_type'])+":"+str(new_sentence['spo_list'][0]['object_type']['@value']))
            # print(str(new_sentence['spo_list'][0]['subject_type'])+":"+str(new_sentence['spo_list'][0]['object_type']['@value']))
            # print(new_sentence['spo_list'][0]['object_type']['@value'])

            sentence_list.append(new_sentence)

    from collections import Counter  # 导包
    print(Counter(re_count))
    # {'患病部位:疾病名称': 1429, '药品名称:疾病名称': 1230, '检查指标:疾病名称': 1027, '分期分型:疾病名称': 1008, '不良反应:药品名称': 627, '非药治疗:疾病名称': 290,
    #  '检查方法:疾病名称': 248, '临床表现:疾病名称': 241, '用药剂量:药品名称': 197, '用药方法:药品名称': 173, '患病病因:疾病名称': 137, '用药频率:药品名称': 112,
    #  '发病机制:疾病名称': 103, '持续时间:药品名称': 57, '临床手术:疾病名称': 30, '不良反应:疾病名称': 2}
    re_count_dict = {'患病部位:疾病名称': 0, '药品名称:疾病名称': 0, '检查指标:疾病名称': 0, '分期分型:疾病名称': 0, '不良反应:药品名称': 0, '非药治疗:疾病名称': 0,
     '检查方法:疾病名称': 0, '临床表现:疾病名称': 0, '用药剂量:药品名称': 0, '用药方法:药品名称': 0, '患病病因:疾病名称': 0, '用药频率:药品名称': 0,
     '发病机制:疾病名称': 0, '持续时间:药品名称': 0, '临床手术:疾病名称': 0, '不良反应:疾病名称': 0}

    random.shuffle(sentence_list)

    n_split_sentence = []
    n_split_sentence_re = []
    for sentence in sentence_list:
        # print(sentence['relation'])
        if(re_count_dict[sentence['relation']]<n_split):
            json_dict = json.dumps(sentence, ensure_ascii=False)
            # new_sentence_list.append(json_dict)
            n_split_sentence.append(json_dict)
            n_split_sentence_re.append(sentence['relation'])
            re_count_dict[sentence['relation']] = re_count_dict[sentence['relation']]+1
    print(Counter(n_split_sentence_re))

    with open(output_train_path,'w+',encoding='utf-8') as f:
        for i in n_split_sentence:
            f.write(str(i)+"\n")
    f.close()


origin_train_path = 'origin_data/CMeIE_train.json'
# output_train_path = 'data_8_split/CMeIE_train.json'
for n_split in [8,16,32,64]:
    if not os.path.exists('data_' + str(n_split) + '_split'):
        os.mkdir('data_' + str(n_split) + '_split')
    output_train_path = 'data_' + str(n_split) + '_split/CMeIE_train.json'
    data_n_split(origin_train_path, output_train_path, n_split)