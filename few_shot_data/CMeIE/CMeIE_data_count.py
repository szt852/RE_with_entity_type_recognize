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
            new_sentence['relation'] = str(new_sentence['spo_list'][0]['predicate'])+":"+str(new_sentence['spo_list'][0]['object_type']['@value'])
            re_count.append(str(new_sentence['spo_list'][0]['predicate'])+":"+str(new_sentence['spo_list'][0]['object_type']['@value']))
            # print(str(new_sentence['spo_list'][0]['subject_type'])+":"+str(new_sentence['spo_list'][0]['object_type']['@value']))
            # print(new_sentence['spo_list'][0]['object_type']['@value'])

            sentence_list.append(new_sentence)

    from collections import Counter  # 导包
    print(Counter(re_count))

    # {'临床表现:症状': 11793, '药物治疗:药物': 4566, '病因:社会学': 2601, '同义词:疾病': 2600, '并发症:疾病': 2019, '病理分型:疾病': 1842,
    #  '实验室检查:检查': 1825, '辅助治疗:其他治疗': 1584, '相关（导致）:疾病': 1441, '影像学检查:检查': 1388, '鉴别诊断:疾病': 1299,
    #  '高危因素:社会学': 1147, '发病部位:部位': 1140, '手术治疗:手术治疗': 886, '相关（转化）:疾病': 722, '多发群体:流行病学': 590, '辅助检查:检查': 570,
    #  '风险评估因素:社会学': 513, '发病率:流行病学': 403, '预防:其他': 373, '相关（症状）:疾病': 363, '组织学检查:检查': 312, '同义词:药物': 294,
    #  '发病年龄:流行病学': 272, '转移部位:部位': 233, '多发地区:流行病学': 230, '预后状况:预后': 206, '阶段:其他': 199, '同义词:检查': 195,
    #  '内窥镜检查:检查': 179, '发病性别倾向:流行病学': 155, '放射治疗:其他治疗': 149, '同义词:社会学': 136, '化疗:其他治疗': 130, '遗传因素:社会学': 128,
    #  '外侵部位:部位': 123, '病史:社会学': 123, '治疗后症状:症状': 114, '筛查:检查': 106, '预后生存率:预后': 79, '多发季节:流行病学': 73,
    #  '死亡率:流行病学': 67, '发病机制:社会学': 51, '传播途径:流行病学': 48, '同义词:其他治疗': 46, '就诊科室:其他': 39, '侵及周围组织转移的症状:症状': 37,
    #  '同义词:手术治疗': 36, '病理生理:社会学': 33, '同义词:症状': 28, '同义词:其他': 15, '同义词:流行病学': 3, '同义词:部位': 2}

    re_count_dict = {'临床表现:症状': 0, '药物治疗:药物': 0, '病因:社会学': 0, '同义词:疾病': 0, '并发症:疾病': 0, '病理分型:疾病': 0,
     '实验室检查:检查': 0, '辅助治疗:其他治疗': 0, '相关（导致）:疾病': 0, '影像学检查:检查': 0, '鉴别诊断:疾病': 0,
     '高危因素:社会学': 0, '发病部位:部位': 0, '手术治疗:手术治疗': 0, '相关（转化）:疾病': 0, '多发群体:流行病学': 0, '辅助检查:检查': 0,
     '风险评估因素:社会学': 0, '发病率:流行病学': 0, '预防:其他': 0, '相关（症状）:疾病': 0, '组织学检查:检查': 0, '同义词:药物': 0,
     '发病年龄:流行病学': 0, '转移部位:部位': 0, '多发地区:流行病学': 0, '预后状况:预后': 0, '阶段:其他': 0, '同义词:检查': 0,
     '内窥镜检查:检查': 0, '发病性别倾向:流行病学': 0, '放射治疗:其他治疗': 0, '同义词:社会学': 0, '化疗:其他治疗': 0, '遗传因素:社会学': 0,
     '外侵部位:部位': 0, '病史:社会学': 0, '治疗后症状:症状': 0, '筛查:检查': 0, '预后生存率:预后': 0, '多发季节:流行病学': 0,
     '死亡率:流行病学': 0, '发病机制:社会学': 0, '传播途径:流行病学': 0, '同义词:其他治疗': 0, '就诊科室:其他': 0, '侵及周围组织转移的症状:症状': 0,
     '同义词:手术治疗': 0, '病理生理:社会学': 0, '同义词:症状': 0, '同义词:其他': 0, '同义词:流行病学': 0, '同义词:部位': 0}

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