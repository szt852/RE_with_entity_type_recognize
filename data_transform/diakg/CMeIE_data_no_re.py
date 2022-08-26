import json
from tqdm import tqdm
import ast
import re
import json
from tqdm import tqdm
import ast

relation_list={'检查方法:疾病名称','临床表现:疾病名称','非药治疗:疾病名称','药品名称:疾病名称','患病部位:疾病名称','患病病因:疾病名称','发病机制:疾病名称','临床手术:疾病名称',
               '分期分型:疾病名称','检查指标:疾病名称','用药频率:药品名称','持续时间:药品名称','用药剂量:药品名称','用药方法:药品名称','不良反应:药品名称','不良反应:疾病名称'}
''' 
所有实体类型( 18 )： {'Frequency', 'Test_items', 'Duration', 'Symptom', 'Pathogenesis', 'Amount', 'ADE', 'Level', 'Class', 'Disease', 'Test_Value', 'Test', 'Operation', 'Drug', 'Method', 'Reason', 'Anatomy', 'Treatment'}
所有关系类型( 16 )： {'Treatment_Disease', 'Operation_Disease', 'ADE_Drug', 'Test_Disease', 'Reason_Disease', 'Frequency_Drug', 'Class_Disease', 'Amount_Drug', 'Test_items_Disease', 'Method_Drug', 'Pathogenesis_Disease', 'Drug_Disease', 'Symptom_Disease', 'Duration_Drug', 'ADE_Disease', 'Anatomy_Disease'}
该文件中的关系数量： 417
所有文件拥有的句子数量： 2292
所有文件拥有的实体数量： 21537
所有文件拥有的关系数量： 8643
筛选后文件拥有的句子数量： 1561

拆分之后sentence总数量 168785
有关系的sentence数量 8643
无关系的sentence数量 160142

疾病关系
1.检查方法:疾病（Test_Disease）
2.临床表现:疾病（Symptom_Disease）
3.非药治疗:疾病（Treatment_Disease）
4.药品名称:疾病（Drug_Disease）
5.患病部位:疾病（Anatomy_Disease）
6.患病病因:疾病（Reason_Disease）
7.发病机制:疾病（Pathogenesis_Disease)
8.临床手术:疾病（Operation_Disese）
9.分期分型:疾病（Class_Disease）
10.检查指标:疾病（Test_Items_Disease）
药物关系
11.用药频率:药品（Frequency_Drug）
12.持续时间:药品（Duration_Drug）
13.用药剂量:药品（Amount_Drug）
14.用药方法:药品（Method_Drug）
15.不良反应:药品（ADE_Drug）
'''


def read_json(path):
    json_data = []
    for line in open(path, 'r', encoding='utf-8'):
        json_data.append(line)
    return json_data

def find_pos(json_content,sentence_list,json_path,no_re_list):
    '''创建实体下标'''

    print("文件",json_path,"拥有的句子数量：",len(json_content))
    # print(json_content)
    for i in tqdm(range(len(json_content))):
        global false,true
        false = False
        true = True
        i_dict = eval(json_content[i])
        # print(type(i_dict))
        # print(len(i_dict['spo_list']))
        temp_no_re_dict = {}
        if(len(i_dict['spo_list'])>1):
            temp_same_sentence_list = []
            for j in i_dict['spo_list']:
                temp_dict = {}
                # print(i_dict['text'])



                # 将无法匹配的字符串更换为可匹配字符串
                j['subject'] = j['subject'].replace('(' , '（')
                j['object']['@value'] = j['object']['@value'].replace('(' , '（')
                i_dict['text'] = i_dict['text'].replace('(' , '（')
                j['subject'] = j['subject'].replace(')' , '）')
                j['object']['@value'] = j['object']['@value'].replace(')' , '）')
                i_dict['text'] = i_dict['text'].replace(')' , '）')

                j['subject'] = j['subject'].replace('[' , '【')
                j['object']['@value'] = j['object']['@value'].replace('[' , '【')
                i_dict['text'] = i_dict['text'].replace('[' , '【')
                j['subject'] = j['subject'].replace(']' , '】')
                j['object']['@value'] = j['object']['@value'].replace(']' , '】')
                i_dict['text'] = i_dict['text'].replace(']' , '】')

                j['subject'] = j['subject'].replace(' ', '')
                j['object']['@value'] = j['object']['@value'].replace(' ', '')
                i_dict['text'] = i_dict['text'].replace(' ', '')

                j['subject'] = j['subject'].replace('+', '加')
                j['object']['@value'] = j['object']['@value'].replace('+', '加')
                i_dict['text'] = i_dict['text'].replace('+', '加')

                j['subject'] = j['subject'].replace('^3', '立方')
                j['object']['@value'] = j['object']['@value'].replace('^3', '立方')
                i_dict['text'] = i_dict['text'].replace('^3', '立方')

                j['subject'] = j['subject'].replace('＜', '<')
                j['object']['@value'] = j['object']['@value'].replace('＜', '<')
                i_dict['text'] = i_dict['text'].replace('＜', '<')

                j['subject'] = j['subject'].replace('＞', '>')
                j['object']['@value'] = j['object']['@value'].replace('＞', '>')
                i_dict['text'] = i_dict['text'].replace('＞', '>')

                j['subject'] = j['subject'].replace('^2', '平方')
                j['object']['@value'] = j['object']['@value'].replace('^2', '平方')
                i_dict['text'] = i_dict['text'].replace('^2', '平方')

                j['subject'] = j['subject'].replace('^9', '9次方')
                j['object']['@value'] = j['object']['@value'].replace('^9', '9次方')
                i_dict['text'] = i_dict['text'].replace('^9', '9次方')

                j['subject'] = j['subject'].replace('*', '星')
                j['object']['@value'] = j['object']['@value'].replace('*', '星')
                i_dict['text'] = i_dict['text'].replace('*', '星')




                # print(j)
                # print(j['subject'])
                # print(j['object']['@value'])
                # if(j['subject'] == "胱抑素C(CysC)"):
                    # print("!!!")

                sub_pos = list(re.search(j['subject'], i_dict['text']).span())
                # obj_pos = re.search(j['object']['@value'], i_dict['text'])
                # obj_pos_group = re.search(j['object']['@value'], i_dict['text']).group()
                obj_pos = list(re.search(j['object']['@value'], i_dict['text']).span())
                # print(sub_pos)
                # print(obj_pos)
                entity_type_list.append(j['subject_type'])
                entity_type_list.append(j['object_type']['@value'])

                h_dict = {}
                t_dict={}
                if(j['predicate'] != '同义词'):
                    h_dict['name']=j['subject']
                    h_dict['pos']=sub_pos
                    h_dict['type']=str(j['subject_type'])
                    t_dict['name'] = j['object']['@value']
                    t_dict['pos'] = obj_pos
                    t_dict['type'] = str(j['object_type']['@value'])
                    # t_dict['root_type'] = j['object_type']['@value']

                    temp_dict['sentence'] = i_dict['text']
                    temp_dict['h'] = h_dict
                    temp_dict['t'] = t_dict
                    temp_dict['relation'] = str(h_dict['type'])+":"+str(t_dict['type'])
                    relation_type_list.append(temp_dict['relation'])

                # elif(j['predicate'] == '同义词'):
                #     h_dict['name'] = j['subject']
                #     h_dict['pos'] = sub_pos
                #     h_dict['type'] = str(j['subject_type'])
                #
                #     t_dict['name'] = j['object']['@value']
                #     t_dict['pos'] = obj_pos
                #     t_dict['type'] = str(j['object_type']['@value'])
                #     # t_dict['root_type'] = j['object_type']['@value']
                #
                #     temp_dict['sentence'] = i_dict['text']
                #     temp_dict['h'] = h_dict
                #     temp_dict['t'] = t_dict
                #     temp_dict['relation'] = str(h_dict['type']) + ":" + str(t_dict['type'])
                #     relation_type_list.append(temp_dict['relation'])

                sentence_list.append(temp_dict)
                temp_same_sentence_list.append(temp_dict)
            if (len(temp_same_sentence_list)>1):
                index_num = list(range(0,len(temp_same_sentence_list)))
                for k in index_num:
                    # print(k)
                    # print(temp_same_sentence_list[k])
                    # print(temp_same_sentence_list[k]['h'])
                    # print(temp_same_sentence_list[k]['t'])
                    # print(temp_same_sentence_list[k]['relation'])
                    index_num.remove(k)
                    for l in index_num:
                        # print(k,l)
                        temp_no_re_dict['sentence'] = temp_same_sentence_list[k]['sentence']
                        temp_no_re_dict['h'] = temp_same_sentence_list[k]['h']
                        temp_no_re_dict['t'] = temp_same_sentence_list[l]['t']
                        temp_no_re_dict['relation'] = str(temp_same_sentence_list[k]['h']['type']) +":"+ str(temp_same_sentence_list[l]['t']['type'])
                        no_re_list.append(temp_no_re_dict)

                    index_num.append(k)
                    index_num.sort(reverse=False)


    # 删除不可能存在的关系
    print("no_re_list的数据量:",len(no_re_list))
    jiangyou_list = []
    for m in tqdm(range(len(no_re_list))):
        if(no_re_list[m]['relation'] in relation_list):
            jiangyou_list.append(no_re_list[m])
    no_re_list = jiangyou_list
    print("删除不可能存在的关系后no_re_list的数据量:", len(no_re_list))

    # 字典列表去重
    unique_list = []
    temp_no_re_list = sorted(no_re_list, key=lambda x: x['sentence'])
    for n in tqdm(range(len(temp_no_re_list))):
        if temp_no_re_list[n] not in unique_list:
            unique_list.append(temp_no_re_list[n])
    no_re_list = unique_list

    # 取差集
    print("去重后关系后no_re_list的数据量:",len(no_re_list))
    # no_re_list = [s for s in no_re_list if s not in sentence_list]
    # no_re_list = list[set(no_re_list)^set(sentence_list)]
    no_re_ok_list = []
    for s in tqdm(range(len(no_re_list))):
        if no_re_list[s] not in sentence_list:
            no_re_ok_list.append(no_re_list[s])
    print("no_re_list与存在关系数据取差集后的的数据量:",len(no_re_ok_list))

    for t in no_re_ok_list:
        t['relation'] = "非:"+str(t['relation'])

    return sentence_list,no_re_ok_list


def fix_pos(sentence_list):
    no_re_re_list=[]
    two_e_sentence_list = []
    for i in tqdm(range(len(sentence_list))):

        # 重新匹配pos
        # print(sentence_list[i]['h']['type'])
        # print(sentence_list[i]['sentence'])
        sub_pos = list(re.search(sentence_list[i]['h']['name'], sentence_list[i]['sentence']).span())
        obj_pos = list(re.search(sentence_list[i]['t']['name'], sentence_list[i]['sentence']).span())
        sentence_list[i]['h']['pos'] = sub_pos
        sentence_list[i]['t']['pos'] = obj_pos

        '''找到两个实体的最开始位置和最后位置'''
        pos_min=min(sentence_list[i]['h']['pos'][0],sentence_list[i]['h']['pos'][1],
              sentence_list[i]['t']['pos'][0],sentence_list[i]['t']['pos'][1])
        pos_max=max(sentence_list[i]['h']['pos'][0],sentence_list[i]['h']['pos'][1],
              sentence_list[i]['t']['pos'][0],sentence_list[i]['t']['pos'][1])
        # print("pos_min:",pos_min,"---pos_max:",pos_max)

        '''向sentence左边搜索逗号、句号'''
        left = pos_min
        while left>=0 and sentence_list[i]['sentence'][left] not in ["，","。"]:
            left=left-1
        left=left+1

        '''向sentence右边搜索逗号，句号'''
        right = pos_max
        while right < len(sentence_list[i]['sentence']) and sentence_list[i]['sentence'][right] not in ["，", "。"]:
            right = right + 1
        # print("left:",left,"---right:",right)
        # print(sentence_list[i]['sentence'][left:right]+"。")

        '''修改pos'''
        sentence_list[i]['h']['pos'] = [sentence_list[i]['h']['pos'][0]-left,sentence_list[i]['h']['pos'][1]-left]
        sentence_list[i]['t']['pos'] = [sentence_list[i]['t']['pos'][0]-left,sentence_list[i]['t']['pos'][1]-left]

        '''建立临时字典'''
        two_e_sentence_dict = {}
        # two_e_sentence_dict['sentence'] = sentence_list[i]['sentence'][left:right] + "。"
        two_e_sentence_dict['token'] = [char for char in sentence_list[i]['sentence'][left:right] + "。"]
        two_e_sentence_dict['h'] = sentence_list[i]['h']
        two_e_sentence_dict['t'] = sentence_list[i]['t']
        two_e_sentence_dict['relation'] = sentence_list[i]['relation']
        # print(two_e_sentence_dict)
        no_re_re_list.append(two_e_sentence_dict['relation'])

        two_e_sentence_list.append(two_e_sentence_dict)

    # print("所有关系(",len(no_re_re_list),")：",no_re_re_list)
    no_re_re_list = set(no_re_re_list)
    print("所有关系(",len(no_re_re_list),")：",no_re_re_list)
    return two_e_sentence_list


'''1.遍历所有文件，将句子分为多个只有一对头尾实体的句子，补充pos'''
entity_type_list=[]
relation_type_list=[]
sentence_list = []
no_re_list = []
dev_json_path='../../datasets/diakg_org_split/CMeIE_dev.json'
test_json_path='../../datasets/diakg_org_split/CMeIE_test.json'
train_json_path='../../datasets/diakg_org_split/CMeIE_train.json'

for path_i in [dev_json_path,test_json_path,train_json_path]:
    tmp_sentence_list = []
    tmp_no_re_list = []
    tmp_sentence_list, tmp_no_re_list = find_pos(read_json(path_i), tmp_sentence_list, path_i, tmp_no_re_list)
    print(path_i, "拥有的句子数量：", len(tmp_sentence_list))
    print(path_i, "所有文件拥有无关系的句子数量：", len(tmp_no_re_list))
    sentence_list = sentence_list + tmp_sentence_list
    no_re_list = no_re_list + tmp_no_re_list

    tmp_dataset = path_i.split('/')[4].split('_')[1].split('.')[0]
    two_e_sentence_no_re_list = fix_pos(tmp_sentence_list + tmp_no_re_list)
    CMeIE_no_re_ok_path = '../../datasets/diakg_org_split/' + tmp_dataset + '.txt'
    with open(CMeIE_no_re_ok_path, 'w+', encoding='utf-8') as f:
        for i in two_e_sentence_no_re_list:
            f.write(str(i) + "\n")
    f.close()


print("所有文件拥有的句子数量：",len(sentence_list))
print("所有文件拥有的无关系句子数量：",len(no_re_list))
relation_type_list = set(relation_type_list)
entity_type_list = set(entity_type_list)
print("所有实体类型(",len(entity_type_list),")：",entity_type_list)
print("所有关系类型(",len(relation_type_list),")：",relation_type_list)

# re_count_dict={}
re_count_list=[]
for tmp_r in sentence_list + no_re_list:
    re_count_list.append(tmp_r['relation'])

from collections import Counter
count=Counter(re_count_list)
print("关系分布统计: ","共",len(count),"种关系：",count)

'''
E:\Anaconda3\envs\torch1.8\python.exe C:/Users/Administrator/nlpprogram/CHRE-PROMPT/data_transform/diakg/CMeIE_data_no_re.py
文件 ../../datasets/diakg_org_split/CMeIE_dev.json 拥有的句子数量： 156
no_re_list的数据量: 9874
删除不可能存在的关系后no_re_list的数据量: 9442
去重后关系后no_re_list的数据量: 114
no_re_list与存在关系数据取差集后的的数据量: 29
../../datasets/diakg_org_split/CMeIE_dev.json 拥有的句子数量： 831
../../datasets/diakg_org_split/CMeIE_dev.json 所有文件拥有无关系的句子数量： 29
所有关系( 24 )： {'药品名称:疾病名称', '临床表现:疾病名称', '非:患病部位:疾病名称', '非:临床表现:疾病名称', '患病病因:疾病名称', '非药治疗:疾病名称', '用药剂量:药品名称', '持续时间:药品名称', '患病部位:疾病名称', '非:不良反应:药品名称', 
'检查方法:疾病名称', '检查指标:疾病名称', '用药方法:药品名称', '非:用药剂量:药品名称', '不良反应:药品名称', '非:非药治疗:疾病名称', '发病机制:疾病名称', '用药频率:药品名称', '非:用药频率:药品名称', '分期分型:疾病名称', '临床手术:疾病名称', 
'非:检查指标:疾病名称', '非:分期分型:疾病名称', '非:不良反应:疾病名称'}

文件 ../../datasets/diakg_org_split/CMeIE_test.json 拥有的句子数量： 157
no_re_list的数据量: 8736
删除不可能存在的关系后no_re_list的数据量: 7922
去重后关系后no_re_list的数据量: 114
no_re_list与存在关系数据取差集后的的数据量: 27
../../datasets/diakg_org_split/CMeIE_test.json 拥有的句子数量： 830
../../datasets/diakg_org_split/CMeIE_test.json 所有文件拥有无关系的句子数量： 27
所有关系( 25 )： {'药品名称:疾病名称', '临床表现:疾病名称', '非:患病部位:疾病名称', '非:用药方法:药品名称', '患病病因:疾病名称', '非药治疗:疾病名称', '用药剂量:药品名称', '非:患病病因:疾病名称', '持续时间:药品名称', '患病部位:疾病名称', 
'非:不良反应:药品名称', '检查方法:疾病名称', '检查指标:疾病名称', '用药方法:药品名称', '非:药品名称:疾病名称', '不良反应:药品名称', '发病机制:疾病名称', '用药频率:药品名称', '非:用药频率:药品名称', '分期分型:疾病名称', '临床手术:疾病名称', 
'非:分期分型:疾病名称', '非:检查指标:疾病名称', '非:临床手术:疾病名称', '非:不良反应:疾病名称'}

文件 ../../datasets/diakg_org_split/CMeIE_train.json 拥有的句子数量： 1248
no_re_list的数据量: 69330
删除不可能存在的关系后no_re_list的数据量: 66084
去重后关系后no_re_list的数据量: 942
no_re_list与存在关系数据取差集后的的数据量: 219
../../datasets/diakg_org_split/CMeIE_train.json 拥有的句子数量： 6650
../../datasets/diakg_org_split/CMeIE_train.json 所有文件拥有无关系的句子数量： 219
所有关系( 30 )： {'药品名称:疾病名称', '临床表现:疾病名称', '非:患病部位:疾病名称', '非:用药方法:药品名称', '患病病因:疾病名称', '非:临床表现:疾病名称', '非药治疗:疾病名称', '用药剂量:药品名称', '非:患病病因:疾病名称', '非:发病机制:疾病名称', 
'持续时间:药品名称', '患病部位:疾病名称', '非:不良反应:药品名称', '检查方法:疾病名称', '检查指标:疾病名称', '用药方法:药品名称', '非:药品名称:疾病名称', '非:用药剂量:药品名称', '非:非药治疗:疾病名称', '不良反应:药品名称', '发病机制:疾病名称', 
'用药频率:药品名称', '非:用药频率:药品名称', '分期分型:疾病名称', '临床手术:疾病名称', '非:分期分型:疾病名称', '非:检查指标:疾病名称', '不良反应:疾病名称', '非:不良反应:疾病名称', '非:检查方法:疾病名称'}

所有文件拥有的句子数量： 8311
所有文件拥有的无关系句子数量： 275

所有实体类型( 16 )： {'临床表现', '患病部位', '用药方法', '发病机制', '疾病名称', '持续时间', '检查方法', '分期分型', '药品名称', '不良反应', '临床手术', '检查指标', '非药治疗', '患病病因', '用药剂量', '用药频率'}
所有关系类型( 16 )： {'药品名称:疾病名称', '临床表现:疾病名称', '用药频率:药品名称', '发病机制:疾病名称', '患病病因:疾病名称', '非药治疗:疾病名称', '分期分型:疾病名称', '用药剂量:药品名称', '持续时间:药品名称', '临床手术:疾病名称', '检查方法:疾病名称', 
'不良反应:疾病名称', '检查指标:疾病名称', '用药方法:药品名称', '患病部位:疾病名称', '不良反应:药品名称'}

关系分布统计:  共 31 种关系： Counter({'患病部位:疾病名称': 1688, '药品名称:疾病名称': 1490, '检查指标:疾病名称': 1242, '分期分型:疾病名称': 1169, '不良反应:药品名称': 756, '非药治疗:疾病名称': 369, '检查方法:疾病名称': 316, 
'临床表现:疾病名称': 294, '用药剂量:药品名称': 230, '用药方法:药品名称': 200, '患病病因:疾病名称': 182, '发病机制:疾病名称': 137, '非:患病部位:疾病名称': 134, '用药频率:药品名称': 128, '持续时间:药品名称': 68, '非:分期分型:疾病名称': 45, 
'临床手术:疾病名称': 40, '非:不良反应:疾病名称': 23, '非:检查指标:疾病名称': 14, '非:不良反应:药品名称': 11, '非:用药剂量:药品名称': 10, '非:非药治疗:疾病名称': 7, '非:检查方法:疾病名称': 7, '非:药品名称:疾病名称': 6, '非:患病病因:疾病名称': 5,
 '非:用药频率:药品名称': 4, '非:用药方法:药品名称': 4, '非:临床表现:疾病名称': 3, '不良反应:疾病名称': 2, '非:临床手术:疾病名称': 1, '非:发病机制:疾病名称': 1})
'''