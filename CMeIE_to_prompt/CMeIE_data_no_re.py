import json
from tqdm import tqdm
import ast
import re
'''
文件 ../datasets/CMeIE_org/CMeIE_dev.jsonl 拥有的句子数量： 3585
no_re_list的数据量: 48506
删除不可能存在的关系后no_re_list的数据量: 47374
去重后关系后no_re_list的数据量: 2126
no_re_list与存在关系数据取差集后的的数据量: 295
拥有的句子数量： 14339
no_re_list的数据量: 201047
删除不可能存在的关系后no_re_list的数据量: 195877
去重后关系后no_re_list的数据量: 9007
no_re_list与存在关系数据取差集后的的数据量: 1440
所有文件拥有的句子数量： 47573

所有实体类型( 53 )： {'辅助检查', '预后状况', '外侵部位', '药物', '病史', '发病部位', '发病性别倾向', '多发季节',
'相关（转化）', '就诊科室', '转移部位', '内窥镜检查', '化疗', '相关（症状）', '组织学检查', '死亡率', '筛查', '遗传因素',
'社会学', '辅助治疗', '检查', '同义词', '阶段', '其他', '症状', '疾病', '影像学检查', '病因', '流行病学', '风险评估因素',
'多发地区', '侵及周围组织转移的症状', '相关（导致）', '放射治疗', '实验室检查', '临床表现', '药物治疗', '发病年龄', '发病率',
'并发症', '高危因素', '手术治疗', '部位', '预后生存率', '传播途径', '预防', '鉴别诊断', '发病机制', '病理分型', '多发群体', 
'治疗后症状', '病理生理', '其他治疗'}

所有关系类型( 53 )： {'疾病名称:性别倾向', '疾病名称:多发季节', '疾病名称:发病年龄', '社会学名:社会学名', '疾病名称:相关转化', 
'疾病名称:治后症状', '疾病名称:预后状况', '手术治疗:手术治疗', '疾病名称:组织检查', '疾病名称:疾病阶段', '疾病名称:传播途径', 
'疾病名称:化疗治疗', '疾病名称:疾病病史', '疾病名称:预防方法', '其他治疗:其他治疗', '疾病名称:相关导致', '身体部位:身体部位', 
'疾病名称:多发群体', '疾病名称:发病机制', '疾病名称:鉴别诊断', '疾病名称:转移部位', '疾病名称:辅助检查', '药物名称:药物名称', 
'疾病名称:死亡概率', '疾病名称:外侵部位', '疾病名称:相关症状', '检查名称:检查名称', '疾病名称:预后生率', '疾病名称:遗传因素', 
'疾病名称:手术治疗', '疾病名称:影像检查', '其他实体:其他实体', '疾病名称:临床表现', '疾病名称:发病概率', '疾病名称:多发地区', 
'疾病名称:疾病病因', '疾病名称:病理生理', '疾病名称:发病部位', '疾病名称:就诊科室', '疾病名称:放射治疗', '疾病名称:内镜检查', 
'疾病症状:疾病症状', '疾病名称:疾病名称', '疾病名称:侵及症状', '疾病名称:并发病症', '疾病名称:病理分型', '疾病名称:辅助治疗', 
'疾病名称:筛查检查', '流行病学:流行病学', '疾病名称:风险因素', '疾病名称:实验检查', '疾病名称:高危因素', '疾病名称:药物治疗'}
'''

relation_list = ['疾病名称:性别倾向', '疾病名称:多发季节', '疾病名称:发病年龄', '社会学名:社会学名', '疾病名称:相关转化',
'疾病名称:治后症状', '疾病名称:预后状况', '手术治疗:手术治疗', '疾病名称:组织检查', '疾病名称:疾病阶段', '疾病名称:传播途径',
'疾病名称:化疗治疗', '疾病名称:疾病病史', '疾病名称:预防方法', '其他治疗:其他治疗', '疾病名称:相关导致', '身体部位:身体部位',
'疾病名称:多发群体', '疾病名称:发病机制', '疾病名称:鉴别诊断', '疾病名称:转移部位', '疾病名称:辅助检查', '药物名称:药物名称',
'疾病名称:死亡概率', '疾病名称:外侵部位', '疾病名称:相关症状', '检查名称:检查名称', '疾病名称:预后生率', '疾病名称:遗传因素',
'疾病名称:手术治疗', '疾病名称:影像检查', '其他实体:其他实体', '疾病名称:临床表现', '疾病名称:发病概率', '疾病名称:多发地区',
'疾病名称:疾病病因', '疾病名称:病理生理', '疾病名称:发病部位', '疾病名称:就诊科室', '疾病名称:放射治疗', '疾病名称:内镜检查',
'疾病症状:疾病症状', '疾病名称:疾病名称', '疾病名称:侵及症状', '疾病名称:并发病症', '疾病名称:病理分型', '疾病名称:辅助治疗',
'疾病名称:筛查检查', '流行病学:流行病学', '疾病名称:风险因素', '疾病名称:实验检查', '疾病名称:高危因素', '疾病名称:药物治疗']

entity_type_dict={
'其他':'其他实体','预防':'预防方法','阶段':'疾病阶段','就诊科室':'就诊科室',
'其他治疗':'其他治疗','辅助治疗':'辅助治疗','化疗':'化疗治疗','放射治疗':'放射治疗',
'手术治疗':'手术治疗',
'检查':'检查名称','实验室检查':'实验检查','影像学检查':'影像检查','辅助检查':'辅助检查','组织学检查':'组织检查','内窥镜检查':'内镜检查','筛查':'筛查检查',
'流行病学':'流行病学','多发群体':'多发群体','发病率':'发病概率','发病年龄':'发病年龄','多发地区':'多发地区','发病性别倾向':'性别倾向','死亡率':'死亡概率','多发季节':'多发季节','传播途径':'传播途径',
'疾病':'疾病名称','并发症':'并发病症','病理分型':'病理分型','相关（导致）':'相关导致','鉴别诊断':'鉴别诊断','相关（转化）':'相关转化','相关（症状）':'相关症状',
'症状':'疾病症状','临床表现':'临床表现','治疗后症状':'治后症状','侵及周围组织转移的症状':'侵及症状',
'社会学':'社会学名','病因':'疾病病因','高危因素':'高危因素','风险评估因素':'风险因素','病史':'疾病病史','遗传因素':'遗传因素','发病机制':'发病机制','病理生理':'病理生理',
'药物':'药物名称','药物治疗':'药物治疗',
'发病部位':'发病部位','转移部位':'转移部位','外侵部位':'外侵部位','部位':'身体部位',
'预后状况':'预后状况','预后生存率':'预后生率'
}


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

                sub_pos = list(re.search(j['subject'], i_dict['text']).span())
                # obj_pos = re.search(j['object']['@value'], i_dict['text'])
                # obj_pos_group = re.search(j['object']['@value'], i_dict['text']).group()
                obj_pos = list(re.search(j['object']['@value'], i_dict['text']).span())
                # print(sub_pos)
                # print(obj_pos)
                entity_type_list.append(j['subject_type'])
                entity_type_list.append(j['predicate'])

                h_dict = {}
                t_dict={}
                if(j['predicate'] != '同义词'):
                    h_dict['name']=j['subject']
                    h_dict['pos']=sub_pos
                    h_dict['type']=entity_type_dict[str(j['subject_type'])]

                    t_dict['name'] = j['object']['@value']
                    t_dict['pos'] = obj_pos
                    t_dict['type'] = entity_type_dict[str(j['predicate'])]
                    # t_dict['root_type'] = j['object_type']['@value']

                    temp_dict['sentence'] = i_dict['text']
                    temp_dict['h'] = h_dict
                    temp_dict['t'] = t_dict
                    temp_dict['relation'] = str(h_dict['type'])+":"+str(t_dict['type'])
                    relation_type_list.append(temp_dict['relation'])

                elif(j['predicate'] == '同义词'):
                    h_dict['name'] = j['subject']
                    h_dict['pos'] = sub_pos
                    h_dict['type'] = entity_type_dict[str(j['subject_type'])]

                    t_dict['name'] = j['object']['@value']
                    t_dict['pos'] = obj_pos
                    t_dict['type'] = entity_type_dict[str(j['object_type']['@value'])]
                    # t_dict['root_type'] = j['object_type']['@value']

                    temp_dict['sentence'] = i_dict['text']
                    temp_dict['h'] = h_dict
                    temp_dict['t'] = t_dict
                    temp_dict['relation'] = str(h_dict['type']) + ":" + str(t_dict['type'])
                    relation_type_list.append(temp_dict['relation'])

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


    return sentence_list,no_re_ok_list

'''1.遍历所有文件，将句子分为多个只有一对头尾实体的句子，补充pos'''
sentence_list=[]
entity_type_list=[]
relation_type_list=[]
no_re_list = []
# for i in range(41):
    # print(i)
dev_json_path='../datasets/CMeIE_org/CMeIE_dev.jsonl'
sentence_list,no_re_list=find_pos(read_json(dev_json_path),sentence_list,dev_json_path,no_re_list)
train_json_path='../datasets/CMeIE_org/CMeIE_train.jsonl'
sentence_list,no_re_list=find_pos(read_json(train_json_path),sentence_list,train_json_path,no_re_list)

# print(sentence_list)
print("所有文件拥有的句子数量：",len(sentence_list))
relation_type_list = set(relation_type_list)
entity_type_list = set(entity_type_list)
# print("所有实体类型(",len(entity_type_list),")：",entity_type_list)
# print("所有关系类型(",len(relation_type_list),")：",relation_type_list)


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
        two_e_sentence_dict['relation'] = sentence_list[i]['relation']+":无"
        # print(two_e_sentence_dict)
        no_re_re_list.append(two_e_sentence_dict['relation'])

        two_e_sentence_list.append(two_e_sentence_dict)

    print("所有关系(",len(no_re_re_list),")：",no_re_re_list)
    no_re_re_list = set(no_re_re_list)
    print("所有关系(",len(no_re_re_list),")：",no_re_re_list)
    return two_e_sentence_list

two_e_sentence_no_re_list = fix_pos(no_re_list)

CMeIE_no_re_ok_path = '../datasets/CMeIE_with_no_re/CMeIE_no_re_ok.txt'
with open(CMeIE_no_re_ok_path, 'w+', encoding='utf-8') as f:
    for i in two_e_sentence_no_re_list:
        f.write(str(i) + "\n")
f.close()