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

    # for s in tqdm(range(len(sentence_list))):
    #     tmp_sentence_dict = {}
    #     # print(sentence_list[s]['sentence'])
    #     tmp_sentence_dict['text'] = sentence_list[s]['sentence']
    #     tmp_sentence_dict['spo_list'] = []
    #     for r in sentence_list[s]['relations']:
    #         tmp_spo_dict = {}
    #         tmp_object_dict = {}
    #         tmp_object_type_dict = {}
    #         tmp_spo_dict['Combined'] = False
    #         tmp_spo_dict['predicate'] = [e['entity_chinese_type'] for e in sentence_list[s]['entities'] if e['entity_id']==r['tail_entity_id']][0]
    #         # tmp_spo_dict['predicate'] = r['relation_chinese_type']
    #         tmp_spo_dict['subject'] = [e['name'] for e in sentence_list[s]['entities'] if e['entity_id']==r['head_entity_id']][0]
    #         tmp_spo_dict['subject_type'] = [e['entity_chinese_type'] for e in sentence_list[s]['entities'] if e['entity_id']==r['head_entity_id']][0]
    #         tmp_object_dict['@value'] = [e['name'] for e in sentence_list[s]['entities'] if e['entity_id']==r['tail_entity_id']][0]
    #         tmp_object_type_dict['@value'] = [e['entity_chinese_type'] for e in sentence_list[s]['entities'] if e['entity_id']==r['tail_entity_id']][0]
    #         tmp_spo_dict['object'] = tmp_object_dict
    #         tmp_spo_dict['object_type'] = tmp_object_type_dict
    #         tmp_sentence_dict['spo_list'].append(tmp_spo_dict)
    #     json_dict = json.dumps(tmp_sentence_dict, ensure_ascii = False)
    #     new_sentence_list.append(json_dict)

    return sentence_list,sum_json_content

'''1.遍历所有文件，构造需要的k，v对，删除只包含一个实体的句子。'''
sentence_list=[]
entity_type_list=[]
relation_type_list=[]
new_sentence_list = []

for i in range(41):
    # print(i)
    json_path='../datasets/diakg_org/'+str(i+1)+'.json'
    sentence_list,sum_json_content = contain_relations(read_json(json_path), sentence_list, json_path, sum_json_content)

# print(sentence_list)
print("所有文件拥有的句子数量：",sum_json_content)
print("所有文件拥有的实体数量：",len(entity_type_list))
print("所有文件拥有的关系数量：",len(relation_type_list))
print("筛选后文件拥有的句子数量：",len(sentence_list))
relation_type_list = set(relation_type_list)
entity_type_list = set(entity_type_list)
print("所有实体类型(",len(entity_type_list),")：",entity_type_list)
print("所有关系类型(",len(relation_type_list),")：",relation_type_list)

def to_CMeIE_type(sentence_list):
    for s in tqdm(range(len(sentence_list))):
        tmp_sentence_dict = {}
        # print(sentence_list[s]['sentence'])
        tmp_sentence_dict['text'] = sentence_list[s]['sentence']
        tmp_sentence_dict['spo_list'] = []
        for r in sentence_list[s]['relations']:
            tmp_spo_dict = {}
            tmp_object_dict = {}
            tmp_object_type_dict = {}
            tmp_spo_dict['Combined'] = False
            tmp_spo_dict['predicate'] = [e['entity_chinese_type'] for e in sentence_list[s]['entities'] if e['entity_id']==r['head_entity_id']][0]
            # tmp_spo_dict['predicate'] = r['relation_chinese_type']
            tmp_spo_dict['subject'] = [e['name'] for e in sentence_list[s]['entities'] if e['entity_id']==r['head_entity_id']][0]
            tmp_spo_dict['subject_type'] = [e['entity_chinese_type'] for e in sentence_list[s]['entities'] if e['entity_id']==r['head_entity_id']][0]
            tmp_object_dict['@value'] = [e['name'] for e in sentence_list[s]['entities'] if e['entity_id']==r['tail_entity_id']][0]
            tmp_object_type_dict['@value'] = [e['entity_chinese_type'] for e in sentence_list[s]['entities'] if e['entity_id']==r['tail_entity_id']][0]
            tmp_spo_dict['object'] = tmp_object_dict
            tmp_spo_dict['object_type'] = tmp_object_type_dict
            tmp_sentence_dict['spo_list'].append(tmp_spo_dict)
        json_dict = json.dumps(tmp_sentence_dict, ensure_ascii = False)
        new_sentence_list.append(json_dict)
    return new_sentence_list

new_sentence_list = to_CMeIE_type(sentence_list)
print("筛选后文件拥有的句子数量new_sentence_list：",len(new_sentence_list))


diakg_no_re_path = '../datasets/diakg/diakg_no_re_CMeIE_type.jsonl'
with open(diakg_no_re_path,'w+',encoding='utf-8') as f:
    for i in new_sentence_list:
        f.write(str(i)+"\n")
f.close()
