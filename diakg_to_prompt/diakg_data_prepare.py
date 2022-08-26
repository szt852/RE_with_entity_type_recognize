import json
from tqdm import tqdm
import ast
entity_dict={'Drug':'药品名称', 'Reason':'患病病因', 'Symptom':'临床表现', 'Operation':'临床手术',
             'Amount':'用药剂量','Test_Value':'检查结果', 'Class':'分期分型', 'Test_items':'检查指标',
             'Anatomy':'患病部位', 'Disease':'疾病名称', 'Level':'疾病等级', 'Pathogenesis':'发病机制',
             'Treatment':'非药治疗', 'Duration':'持续时间', 'Frequency':'用药频率', 'Method':'用药方法',
             'Test':'检查方法', 'ADE':'不良反应'}
relation_dict={'Test_Disease':'检查方法:疾病名称','Symptom_Disease':'临床表现:疾病名称','Treatment_Disease':'非药治疗:疾病名称',
           'Drug_Disease':'药品名称:疾病名称','Anatomy_Disease':'患病部位:疾病名称','Reason_Disease':'患病病因:疾病名称',
           'Pathogenesis_Disease':'发病机制:疾病名称','Operation_Disease':'临床手术:疾病名称','Class_Disease':'分期分型:疾病名称',
           'Test_items_Disease':'检查指标:疾病名称','Frequency_Drug':'用药频率:药品名称','Duration_Drug':'持续时间:药品名称',
           'Amount_Drug':'用药剂量:药品名称','Method_Drug':'用药方法:药品名称','ADE_Drug':'不良反应:药品名称',
            'ADE_Disease':'不良反应:疾病名称'}
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
sum_json_content=0

def read_json(path):
    return json.load(open(path,'r',encoding="utf-8"))['paragraphs']

def contain_relations(json_content, sentence_list, json_path, sum_json_content):
    print("文件",json_path,"拥有的句子数量：",len(json_content))
    sum_json_content = sum_json_content + len(json_content)
    # print(json_content)
    re_sum = 0
    for i in json_content:
        paragraphItems=i.items()
        # print(len([v for k,v in paragraphItems if k == 'sentences'][0]))
        sentence_num = len([v for k,v in paragraphItems if k == 'sentences'][0])
        for s in range(sentence_num):
            # print(s)
            sentenceItems = [v for k, v in paragraphItems if k == 'sentences'][0][s].items()
            # print(len(sentenceItems))
            temp_dict = {}
            add = 1
            for sentenceItem in sentenceItems:
                if (str(sentenceItem[0])=='sentence'):
                    temp_dict['sentence']=sentenceItem[1]
                elif (str(sentenceItem[0])=='entities'):
                    if (len(sentenceItem[1])<2):  # 不添加只有一个实体的句子
                        add=0
                        continue
                    temp_dict['entities_num'] = len(sentenceItem[1])
                    temp_dict['potential_relations_num'] = int((len(sentenceItem[1]) * (len(sentenceItem[1]) - 1)) / 2)  # 潜在的可能关系数量是e*(e-1）/2
                    temp_dict['entities'] = sentenceItem[1]
                    for e in temp_dict['entities']:
                        e['pos']=[e['start_idx'],e['end_idx']]
                        e['name']=e['entity']
                        e['entity_chinese_type'] = entity_dict[str(e['entity_type'])]
                        entity_type_list.append(str(e['entity_type']))
                elif (str(sentenceItem[0])=='relations'):
                    # print(len(sentenceItem[1]))
                    re_sum = re_sum + len(sentenceItem[1])
                    # print(re_sum)
                    if (len(sentenceItem[1])<1):  # 不添加只有一个实体的句子
                        add=0
                        continue
                    temp_dict['relations_num'] = len(sentenceItem[1])
                    temp_dict['relations'] = sentenceItem[1]
                    for r in temp_dict['relations']:
                        r['relation_chinese_type']=relation_dict[str(r['relation_type'])]
                        relation_type_list.append(str(r['relation_type']))

            if(add==1):
                sentence_list.append(temp_dict)
    # print(sentence_list)
    # print(len(sentence_list))
    print("该文件中的关系数量：",re_sum)
    return sentence_list,sum_json_content

'''1.遍历所有文件，构造需要的k，v对，删除只包含一个实体的句子。'''
sentence_list=[]
entity_type_list=[]
relation_type_list=[]
for i in range(41):
    # print(i)
    json_path='../datasets/diakg_org/'+str(i+1)+'.json'
    sentence_list,sum_json_content=contain_relations(read_json(json_path),sentence_list,json_path,sum_json_content)

# print(sentence_list)
print("所有文件拥有的句子数量：",sum_json_content)
print("所有文件拥有的实体数量：",len(entity_type_list))
print("所有文件拥有的关系数量：",len(relation_type_list))
print("筛选后文件拥有的句子数量：",len(sentence_list))
relation_type_list = set(relation_type_list)
entity_type_list = set(entity_type_list)
print("所有实体类型(",len(entity_type_list),")：",entity_type_list)
print("所有关系类型(",len(relation_type_list),")：",relation_type_list)


'''2.将一个长句转变为多个只包含两个实体的句子。'''
def make_2e_sentence(sentence_list):
    two_e_sentence_list_have_re=[]
    two_e_sentence_list_no_re=[]
    two_e_sentence_list=[]
    no_relation_num=0
    relation_num=0
    re_sum = 0
    for i in tqdm(range(len(sentence_list))):
        # print(sentence_list[i]['relations'])
        # print(len(sentence_list[i]['relations']))
        re_sum = re_sum+len(sentence_list[i]['relations'])
        # print("实体数量：", sentence_list[i]['entities_num'])
        # print("潜在关系数量：",sentence_list[i]['potential_relations_num'])
        # print("真实存在关系数量：",sentence_list[i]['relations_num'])
        # print(sentence_list[i])
        for j in range(sentence_list[i]['entities_num']-1):
            for k in range(j+1,sentence_list[i]['entities_num']):
                # print("j----",sentence_list[i]['entities'][j],"\n k----",sentence_list[i]['entities'][k])

                '''找到两个实体的最开始位置和最后位置'''
                pos_min=min(sentence_list[i]['entities'][j]['start_idx'],sentence_list[i]['entities'][j]['end_idx'],
                      sentence_list[i]['entities'][k]['start_idx'],sentence_list[i]['entities'][k]['end_idx'])
                pos_max=max(sentence_list[i]['entities'][j]['start_idx'], sentence_list[i]['entities'][j]['end_idx'],
                          sentence_list[i]['entities'][k]['start_idx'], sentence_list[i]['entities'][k]['end_idx'])
                # print("pos_min:",pos_min,"---pos_max:",pos_max)

                '''向sentence左边搜索逗号，句号(可以暂不实现)'''
                left = pos_min
                while left>=0 and sentence_list[i]['sentence'][left] not in ["，","。"]:
                    left=left-1
                left=left+1

                '''向sentence右边搜索逗号，句号(可以暂不实现)'''
                right = pos_max
                while right < len(sentence_list[i]['sentence']) and sentence_list[i]['sentence'][right] not in ["，", "。"]:
                    right = right + 1
                # print("left:",left,"---right:",right)
                # print(sentence_list[i]['sentence'][left:right]+"。")

                '''建立临时字典'''
                two_e_sentence_dict = {}
                # two_e_sentence_dict['sentence'] = sentence_list[i]['sentence'][left:right] + "。"
                two_e_sentence_dict['token'] = [char for char in sentence_list[i]['sentence'][left:right] + "。"]

                '''区分头实体和尾实体   修改pos'''
                temp_dict1={}
                temp_dict2={}
                temp_dict1['name'] = sentence_list[i]['entities'][j]['entity']
                # temp_dict1['pos'] = sentence_list[i]['entities'][j]['pos']
                temp_dict1['pos'] = [sentence_list[i]['entities'][j]['pos'][0] - left,sentence_list[i]['entities'][j]['pos'][1] - left]
                temp_dict1['type'] = sentence_list[i]['entities'][j]['entity_chinese_type']
                temp_dict1['id'] = sentence_list[i]['entities'][j]['entity_id']

                temp_dict2['name'] = sentence_list[i]['entities'][k]['entity']
                # temp_dict2['pos'] = sentence_list[i]['entities'][k]['pos']
                temp_dict2['pos'] = [sentence_list[i]['entities'][k]['pos'][0] - left,sentence_list[i]['entities'][k]['pos'][1] - left]
                temp_dict2['type'] = sentence_list[i]['entities'][k]['entity_chinese_type']
                temp_dict2['id'] = sentence_list[i]['entities'][k]['entity_id']

                '''如果j实体属于['疾病名称' ,'药品名称']，则j实体为h，k实体为t。'''
                if(sentence_list[i]['entities'][j]['entity_chinese_type'] == '疾病名称'and
                    sentence_list[i]['entities'][k]['entity_chinese_type'] != '疾病名称'):
                    two_e_sentence_dict['h'] = temp_dict1
                    two_e_sentence_dict['t'] = temp_dict2
                elif (sentence_list[i]['entities'][j]['entity_chinese_type'] == '药品名称' and
                        sentence_list[i]['entities'][k]['entity_chinese_type'] not in ['药品名称','疾病名称']):
                    two_e_sentence_dict['h'] = temp_dict1
                    two_e_sentence_dict['t'] = temp_dict2
                else:
                    two_e_sentence_dict['t'] = temp_dict1
                    two_e_sentence_dict['h'] = temp_dict2

                '''在字典中加入关系'''
                '''先将该句可能出现的关系做成一个字典'''
                temp_relation_dict={}
                for id in sentence_list[i]['relations']:
                    temp_relation_dict[str([id['head_entity_id'],id['tail_entity_id']])]=str(id['relation_chinese_type'])

                kv_flag=0
                for key,value in temp_relation_dict.items():
                    # print(key,'-----',value)
                    # print(ast.literal_eval(key))
                    if(sentence_list[i]['entities'][j]['entity_id'] in ast.literal_eval(key) and sentence_list[i]['entities'][k]['entity_id'] in ast.literal_eval(key)):
                        two_e_sentence_dict['relation'] = value
                        two_e_sentence_list_have_re.append(two_e_sentence_dict)  # 加入有关系数据列表
                        relation_num = relation_num + 1
                        kv_flag=1
                        two_e_sentence_list.append(two_e_sentence_dict)  #
                        if (len(two_e_sentence_dict['token']) > 256):
                            print(two_e_sentence_dict)
                        break
                if(kv_flag==0):
                    if (len(two_e_sentence_dict['token']) < 256):  # 不加入过长的无关系数据
                        two_e_sentence_dict['relation'] = '没有关系'
                        two_e_sentence_list_no_re.append(two_e_sentence_dict)  # 加入没有关系数据列表
                        no_relation_num = no_relation_num+1
                        two_e_sentence_list.append(two_e_sentence_dict)  #


                # print("two_e_sentence_dict---",two_e_sentence_dict)
                # two_e_sentence_list.append(two_e_sentence_dict)

    return two_e_sentence_list,relation_num,no_relation_num,two_e_sentence_list_have_re,two_e_sentence_list_no_re


two_e_sentence_list,relation_num,no_relation_num,two_e_sentence_list_have_re,two_e_sentence_list_no_re = make_2e_sentence(sentence_list)
print("拆分之后sentence总数量",len(two_e_sentence_list))
print("有关系的sentence数量",relation_num)
print("无关系的sentence数量",no_relation_num)

diakg_all_path = '../datasets/diakg/diakg_all.txt'
with open(diakg_all_path,'w+',encoding='utf-8') as f:
    for i in two_e_sentence_list:
        f.write(str(i)+"\n")
f.close()

diakg_have_re_path = '../datasets/diakg/diakg_have_re.txt'
with open(diakg_have_re_path,'w+',encoding='utf-8') as f:
    for i in two_e_sentence_list_have_re:
        f.write(str(i)+"\n")
f.close()

diakg_no_re_path = '../datasets/diakg/diakg_no_re.txt'
with open(diakg_no_re_path,'w+',encoding='utf-8') as f:
    for i in two_e_sentence_list_no_re:
        f.write(str(i)+"\n")
f.close()