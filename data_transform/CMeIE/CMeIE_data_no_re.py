import json
from tqdm import tqdm
import ast
import re
'''
文件 ../datasets/CMeIE_org_split/dev.txt 拥有的句子数量： 1792
no_re_list的数据量: 25874
删除不可能存在的关系后no_re_list的数据量: 25256
去重后关系后no_re_list的数据量: 1087
no_re_list与存在关系数据取差集后的的数据量: 152
../datasets/CMeIE_org_split/dev.txt 拥有的句子数量： 4854
../datasets/CMeIE_org_split/dev.txt 所有文件拥有无关系的句子数量： 152
所有关系( 52 )： {'疾病名称:病理生理:无', '疾病名称:遗传因素:无', '疾病名称:筛查检查:无', '检查名称:检查名称:无', '疾病名称:辅助检查:无',
 '疾病名称:预防方法:无', '疾病名称:辅助治疗:无', '疾病名称:并发病症:无', '疾病名称:死亡概率:无', '疾病名称:影像检查:无', '疾病名称:临床表现:无', 
 '疾病名称:化疗治疗:无', '疾病名称:治后症状:无', '疾病名称:病理分型:无', '疾病名称:相关导致:无', '疾病名称:药物治疗:无', '疾病名称:高危因素:无', 
 '疾病名称:鉴别诊断:无', '疾病症状:疾病症状:无', '疾病名称:相关症状:无', '疾病名称:转移部位:无', '药物名称:药物名称:无', '疾病名称:相关转化:无', 
 '疾病名称:预后生率:无', '社会学名:社会学名:无', '疾病名称:多发群体:无', '疾病名称:发病年龄:无', '疾病名称:侵及症状:无', '身体部位:身体部位:无', 
 '疾病名称:疾病名称:无', '疾病名称:手术治疗:无', '疾病名称:风险因素:无', '疾病名称:就诊科室:无', '疾病名称:组织检查:无', '其他实体:其他实体:无', 
 '疾病名称:外侵部位:无', '其他治疗:其他治疗:无', '疾病名称:疾病病史:无', '疾病名称:多发季节:无', '疾病名称:发病概率:无', '疾病名称:实验检查:无',
  '疾病名称:预后状况:无', '疾病名称:多发地区:无', '疾病名称:发病机制:无', '疾病名称:疾病阶段:无', '疾病名称:性别倾向:无', '疾病名称:放射治疗:无',
   '疾病名称:疾病病因:无', '疾病名称:内镜检查:无', '疾病名称:传播途径:无', '手术治疗:手术治疗:无', '疾病名称:发病部位:无'}
 
 
文件 ../datasets/CMeIE_org_split/test.txt 拥有的句子数量： 1793
no_re_list的数据量: 23010
删除不可能存在的关系后no_re_list的数据量: 22576
去重后关系后no_re_list的数据量: 1083
no_re_list与存在关系数据取差集后的的数据量: 137
../datasets/CMeIE_org_split/test.txt 拥有的句子数量： 4580
../datasets/CMeIE_org_split/test.txt 所有文件拥有无关系的句子数量： 137
所有关系( 51 )： {'疾病名称:病理生理:无', '疾病名称:筛查检查:无', '疾病名称:遗传因素:无', '手术治疗:手术治疗:无', '检查名称:检查名称:无',
 '疾病名称:辅助检查:无', '疾病名称:预防方法:无', '疾病名称:并发病症:无', '疾病名称:辅助治疗:无', '疾病名称:死亡概率:无', '疾病名称:影像检查:无', 
 '疾病名称:临床表现:无', '疾病名称:化疗治疗:无', '疾病名称:治后症状:无', '疾病名称:病理分型:无', '疾病名称:相关导致:无', '疾病名称:药物治疗:无', 
 '疾病名称:高危因素:无', '疾病名称:鉴别诊断:无', '疾病症状:疾病症状:无', '疾病名称:相关症状:无', '疾病名称:转移部位:无', '药物名称:药物名称:无', 
 '疾病名称:相关转化:无', '疾病名称:预后生率:无', '社会学名:社会学名:无', '疾病名称:多发群体:无', '疾病名称:发病年龄:无', '身体部位:身体部位:无', 
 '疾病名称:侵及症状:无', '疾病名称:疾病名称:无', '疾病名称:手术治疗:无', '疾病名称:风险因素:无', '疾病名称:就诊科室:无', '疾病名称:组织检查:无', 
 '疾病名称:外侵部位:无', '其他治疗:其他治疗:无', '疾病名称:疾病病史:无', '疾病名称:多发季节:无', '疾病名称:发病概率:无', '疾病名称:实验检查:无',
  '疾病名称:预后状况:无', '疾病名称:发病机制:无', '疾病名称:疾病阶段:无', '疾病名称:性别倾向:无', '疾病名称:放射治疗:无', '疾病名称:疾病病因:无', 
  '疾病名称:内镜检查:无', '疾病名称:传播途径:无', '疾病名称:多发地区:无', '疾病名称:发病部位:无'}

文件 ../datasets/CMeIE_org_split/train.txt 拥有的句子数量： 14339
no_re_list的数据量: 200374
删除不可能存在的关系后no_re_list的数据量: 195124
去重后关系后no_re_list的数据量: 8667
no_re_list与存在关系数据取差集后的的数据量: 1151
../datasets/CMeIE_org_split/train.txt 拥有的句子数量： 38139
../datasets/CMeIE_org_split/train.txt 所有文件拥有无关系的句子数量： 1151
所有关系( 53 )： {'疾病名称:病理生理:无', '疾病名称:遗传因素:无', '疾病名称:筛查检查:无', '手术治疗:手术治疗:无', '检查名称:检查名称:无', 
'疾病名称:辅助检查:无', '疾病名称:预防方法:无', '疾病名称:辅助治疗:无', '疾病名称:并发病症:无', '疾病名称:化疗治疗:无', '疾病名称:影像检查:无',
 '疾病名称:临床表现:无', '疾病名称:死亡概率:无', '疾病名称:治后症状:无', '疾病名称:病理分型:无', '疾病名称:相关导致:无', '疾病名称:药物治疗:无', 
 '疾病名称:鉴别诊断:无', '疾病名称:高危因素:无', '疾病症状:疾病症状:无', '疾病名称:转移部位:无', '疾病名称:相关症状:无', '药物名称:药物名称:无', 
 '疾病名称:相关转化:无', '疾病名称:预后生率:无', '社会学名:社会学名:无', '疾病名称:多发群体:无', '疾病名称:发病年龄:无', '疾病名称:侵及症状:无', 
 '身体部位:身体部位:无', '疾病名称:疾病名称:无', '流行病学:流行病学:无', '疾病名称:手术治疗:无', '疾病名称:风险因素:无', '疾病名称:就诊科室:无', 
 '疾病名称:组织检查:无', '其他实体:其他实体:无', '疾病名称:外侵部位:无', '其他治疗:其他治疗:无', '疾病名称:疾病病史:无', '疾病名称:多发季节:无', 
 '疾病名称:发病概率:无', '疾病名称:预后状况:无', '疾病名称:实验检查:无', '疾病名称:发病机制:无', '疾病名称:性别倾向:无', '疾病名称:疾病阶段:无', 
 '疾病名称:放射治疗:无', '疾病名称:疾病病因:无', '疾病名称:内镜检查:无', '疾病名称:传播途径:无', '疾病名称:多发地区:无', '疾病名称:发病部位:无'}

所有文件拥有的句子数量： 47573
所有文件拥有的无关系句子数量： 1440

所有实体类型( 53 )： {'放射治疗', '鉴别诊断', '就诊科室', '药物', '病理分型', '内窥镜检查', '发病机制', '治疗后症状', '多发地区', '辅助治疗', '疾病', 
'并发症', '实验室检查', '症状', '风险评估因素', '筛查', '检查', '病史', '多发群体', '相关（转化）', '病因', '相关（导致）', '同义词', '手术治疗', '其他治疗', 
'相关（症状）', '部位', '高危因素', '转移部位', '外侵部位', '预防', '组织学检查', '预后生存率', '预后状况', '死亡率', '社会学', '病理生理', '辅助检查', '药物治疗', 
'发病部位', '发病性别倾向', '发病率', '发病年龄', '流行病学', '影像学检查', '化疗', '遗传因素', '侵及周围组织转移的症状', '其他', '阶段', '临床表现', '多发季节', '传播途径'}

所有关系类型( 53 )： {'疾病名称:预后状况', '疾病名称:放射治疗', '疾病名称:发病机制', '疾病名称:病理生理', '疾病名称:发病部位', '疾病名称:影像检查',
 '疾病名称:疾病阶段', '疾病名称:多发群体', '疾病名称:化疗治疗', '流行病学:流行病学', '药物名称:药物名称', '疾病名称:相关症状', '疾病名称:转移部位', 
 '疾病名称:手术治疗', '疾病名称:性别倾向', '疾病名称:辅助治疗', '疾病名称:疾病病史', '手术治疗:手术治疗', '疾病名称:治后症状', '疾病名称:内镜检查', 
 '疾病名称:高危因素', '疾病名称:就诊科室', '疾病名称:实验检查', '疾病名称:侵及症状', '社会学名:社会学名', '身体部位:身体部位', '疾病名称:药物治疗',
  '疾病名称:死亡概率', '其他实体:其他实体', '疾病症状:疾病症状', '检查名称:检查名称', '疾病名称:发病年龄', '疾病名称:临床表现', '疾病名称:鉴别诊断',
   '疾病名称:病理分型', '疾病名称:预防方法', '其他治疗:其他治疗', '疾病名称:筛查检查', '疾病名称:相关导致', '疾病名称:发病概率', '疾病名称:风险因素',
    '疾病名称:组织检查', '疾病名称:遗传因素', '疾病名称:多发地区', '疾病名称:疾病病因', '疾病名称:预后生率', '疾病名称:多发季节', '疾病名称:并发病症', 
    '疾病名称:疾病名称', '疾病名称:外侵部位', '疾病名称:传播途径', '疾病名称:相关转化', '疾病名称:辅助检查'}


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
        if(len(i_dict['spo_list'])>=1):
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

    for t in no_re_ok_list:
        t['relation'] = str(t['relation'])+":无"

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
dev_json_path='../../datasets/CMeIE_org_split/CMeIE_dev.json'
test_json_path='../../datasets/CMeIE_org_split/CMeIE_test.json'
train_json_path='../../datasets/CMeIE_org_split/CMeIE_train.json'

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
    CMeIE_no_re_ok_path = '../../datasets/CMeIE_org_split/' + tmp_dataset + '.txt'
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

re_count_list=[]
for tmp_r in sentence_list + no_re_list:
    re_count_list.append(tmp_r['relation'])

from collections import Counter
count=Counter(re_count_list)
print("关系分布统计: ","共",len(count),"种关系：",count)


'''
文件 ../../datasets/CMeIE_org_split/CMeIE_dev.json 拥有的句子数量： 1792
no_re_list的数据量: 25874
删除不可能存在的关系后no_re_list的数据量: 25256
去重后关系后no_re_list的数据量: 1087
no_re_list与存在关系数据取差集后的的数据量: 152
../../datasets/CMeIE_org_split/CMeIE_dev.json 拥有的句子数量： 5520
../../datasets/CMeIE_org_split/CMeIE_dev.json 所有文件拥有无关系的句子数量： 152
所有关系( 78 )： {'疾病名称:疾病阶段', '药物名称:药物名称', '疾病名称:临床表现', '疾病名称:影像检查:无', '社会学名:社会学名', '身体部位:身体部位', '疾病名称:治后症状', '疾病名称:发病部位:无',
 '疾病名称:外侵部位', '疾病名称:传播途径', '疾病名称:相关转化', '疾病名称:疾病病因', '疾病名称:相关转化:无', '疾病名称:手术治疗:无', '疾病名称:疾病名称', '疾病名称:辅助检查:无', '疾病名称:发病年龄:无',
  '疾病名称:手术治疗', '疾病名称:临床表现:无', '疾病名称:并发病症', '疾病名称:死亡概率', '身体部位:身体部位:无', '其他实体:其他实体', '疾病症状:疾病症状', '疾病名称:高危因素', '疾病名称:疾病病史:无', 
  '疾病名称:风险因素', '疾病名称:实验检查', '疾病名称:发病概率:无', '疾病名称:放射治疗', '疾病名称:实验检查:无', '药物名称:药物名称:无', '疾病名称:性别倾向', '疾病名称:筛查检查', '疾病名称:组织检查',
   '疾病名称:相关导致:无', '疾病名称:疾病病史', '疾病名称:并发病症:无', '疾病名称:药物治疗:无', '疾病名称:鉴别诊断:无', '疾病名称:转移部位', '疾病名称:化疗治疗', '疾病名称:遗传因素', '疾病名称:辅助治疗',
    '疾病名称:辅助治疗:无', '疾病名称:影像检查', '疾病名称:相关症状:无', '疾病名称:多发群体', '其他治疗:其他治疗', '疾病名称:组织检查:无', '疾病名称:预后生率', '手术治疗:手术治疗:无', '疾病名称:内镜检查', 
    '疾病名称:辅助检查', '疾病名称:侵及症状', '检查名称:检查名称', '疾病名称:发病年龄', '疾病名称:疾病病因:无', '疾病名称:病理分型', '疾病名称:多发季节', '疾病名称:预防方法', '疾病名称:发病概率', '疾病名称:疾病名称:无', 
    '手术治疗:手术治疗', '疾病名称:鉴别诊断', '疾病名称:相关导致', '疾病名称:多发群体:无', '疾病名称:发病部位', '疾病名称:高危因素:无', '疾病名称:多发地区', '疾病名称:病理生理', '疾病名称:就诊科室', '疾病名称:相关症状', 
    '检查名称:检查名称:无', '疾病名称:发病机制', '疾病名称:药物治疗', '疾病名称:预后状况', '疾病名称:病理分型:无'}
    
文件 ../../datasets/CMeIE_org_split/CMeIE_test.json 拥有的句子数量： 1793
no_re_list的数据量: 23010
删除不可能存在的关系后no_re_list的数据量: 22576
去重后关系后no_re_list的数据量: 1083
no_re_list与存在关系数据取差集后的的数据量: 137
../../datasets/CMeIE_org_split/CMeIE_test.json 拥有的句子数量： 5260
../../datasets/CMeIE_org_split/CMeIE_test.json 所有文件拥有无关系的句子数量： 137
所有关系( 75 )： {'疾病名称:相关导致:无', '疾病名称:疾病阶段', '疾病名称:疾病病史', '疾病名称:侵及症状', '药物名称:药物名称', '疾病名称:临床表现', '疾病名称:并发病症:无', '疾病名称:影像检查:无', 
'检查名称:检查名称', '疾病名称:发病年龄', '疾病名称:高危因素', '社会学名:社会学名', '疾病症状:疾病症状', '疾病名称:疾病病因:无', '疾病名称:病理分型', '疾病名称:辅助检查', '疾病名称:药物治疗:无', 
'疾病名称:多发季节', '疾病名称:鉴别诊断:无', '疾病名称:预防方法', '疾病名称:转移部位', '疾病名称:预后生率:无', '其他治疗:其他治疗:无', '疾病名称:风险因素', '疾病名称:化疗治疗', '身体部位:身体部位', 
'疾病名称:发病概率', '疾病名称:治后症状', '疾病名称:发病部位:无', '疾病名称:实验检查', '疾病名称:疾病名称:无', '疾病名称:发病概率:无', '疾病名称:外侵部位', '疾病名称:传播途径', '疾病名称:遗传因素', 
'疾病名称:辅助治疗', '疾病名称:相关转化', '疾病名称:疾病病因', '疾病名称:辅助治疗:无', '疾病名称:相关转化:无', '手术治疗:手术治疗', '疾病名称:影像检查', '疾病名称:放射治疗', '疾病名称:鉴别诊断',
 '疾病名称:相关导致', '疾病名称:实验检查:无', '疾病名称:手术治疗:无', '药物名称:药物名称:无', '疾病名称:多发群体:无', '疾病名称:筛查检查:无', '疾病名称:发病部位', '疾病名称:疾病名称', '疾病名称:性别倾向',
  '疾病名称:辅助检查:无', '疾病名称:高危因素:无', '疾病名称:手术治疗', '疾病名称:筛查检查', '疾病名称:多发群体', '其他治疗:其他治疗', '疾病名称:临床表现:无', '疾病名称:多发地区', '疾病名称:预后生率', 
  '疾病名称:病理生理', '疾病名称:并发病症', '手术治疗:手术治疗:无', '疾病名称:组织检查', '疾病名称:内镜检查', '疾病名称:死亡概率', '疾病名称:相关症状', '疾病名称:就诊科室', '检查名称:检查名称:无', 
  '疾病名称:发病机制', '疾病名称:药物治疗', '疾病名称:预后状况', '疾病名称:病理分型:无'}
  
文件 ../../datasets/CMeIE_org_split/CMeIE_train.json 拥有的句子数量： 14339
no_re_list的数据量: 200374
删除不可能存在的关系后no_re_list的数据量: 195124
去重后关系后no_re_list的数据量: 8667
no_re_list与存在关系数据取差集后的的数据量: 1151
../../datasets/CMeIE_org_split/CMeIE_train.json 拥有的句子数量： 43506
../../datasets/CMeIE_org_split/CMeIE_train.json 所有文件拥有无关系的句子数量： 1151
所有关系( 99 )： {'疾病名称:疾病阶段', '疾病名称:侵及症状:无', '药物名称:药物名称', '疾病名称:临床表现', '疾病名称:影像检查:无', '社会学名:社会学名', '疾病症状:疾病症状:无',
 '疾病名称:多发季节:无', '疾病名称:预后状况:无', '身体部位:身体部位', '社会学名:社会学名:无', '疾病名称:治后症状', '疾病名称:发病部位:无', '疾病名称:外侵部位', '疾病名称:传播途径',
  '疾病名称:相关转化', '疾病名称:疾病病因', '疾病名称:相关转化:无', '疾病名称:手术治疗:无', '疾病名称:疾病名称', '疾病名称:发病年龄:无', '疾病名称:辅助检查:无', '疾病名称:手术治疗', 
  '疾病名称:风险因素:无', '疾病名称:临床表现:无', '疾病名称:预防方法:无', '疾病名称:并发病症', '疾病名称:死亡概率', '疾病名称:治后症状:无', '其他实体:其他实体', '疾病症状:疾病症状', 
  '疾病名称:高危因素', '疾病名称:疾病病史:无', '其他实体:其他实体:无', '疾病名称:风险因素', '疾病名称:实验检查', '疾病名称:发病概率:无', '疾病名称:放射治疗', '疾病名称:实验检查:无', 
  '药物名称:药物名称:无', '疾病名称:性别倾向', '疾病名称:传播途径:无', '疾病名称:筛查检查', '流行病学:流行病学', '疾病名称:组织检查', '疾病名称:相关导致:无', '疾病名称:疾病病史', 
  '疾病名称:并发病症:无', '疾病名称:药物治疗:无', '疾病名称:鉴别诊断:无', '疾病名称:转移部位', '其他治疗:其他治疗:无', '疾病名称:预后生率:无', '疾病名称:化疗治疗', '疾病名称:遗传因素:无',
   '疾病名称:遗传因素', '疾病名称:辅助治疗', '疾病名称:内镜检查:无', '疾病名称:辅助治疗:无', '疾病名称:影像检查', '疾病名称:性别倾向:无', '疾病名称:相关症状:无', '疾病名称:多发群体',
    '其他治疗:其他治疗', '疾病名称:组织检查:无', '疾病名称:预后生率', '手术治疗:手术治疗:无', '疾病名称:内镜检查', '疾病名称:辅助检查', '疾病名称:侵及症状', '检查名称:检查名称', 
    '疾病名称:发病年龄', '疾病名称:疾病病因:无', '疾病名称:病理分型', '疾病名称:多发季节', '疾病名称:预防方法', '疾病名称:转移部位:无', '疾病名称:发病概率', '疾病名称:疾病名称:无', 
    '手术治疗:手术治疗', '疾病名称:鉴别诊断', '疾病名称:相关导致', '疾病名称:多发群体:无', '疾病名称:筛查检查:无', '疾病名称:发病部位', '疾病名称:发病机制:无', '疾病名称:高危因素:无', 
    '疾病名称:外侵部位:无', '疾病名称:多发地区', '疾病名称:病理生理', '疾病名称:多发地区:无', '疾病名称:疾病阶段:无', '疾病名称:相关症状', '疾病名称:就诊科室', '检查名称:检查名称:无', 
    '疾病名称:发病机制', '疾病名称:药物治疗', '疾病名称:预后状况', '疾病名称:病理分型:无'}
所有文件拥有的句子数量： 54286
所有文件拥有的无关系句子数量： 1440

所有实体类型( 53 )： {'发病性别倾向', '并发症', '筛查', '多发群体', '转移部位', '手术治疗', '其他', '预后生存率', '相关（转化）', '治疗后症状', '预防', '多发地区', '化疗', '发病机制', 
'外侵部位', '辅助检查', '病史', '实验室检查', '侵及周围组织转移的症状', '相关（导致）', '发病部位', '病理生理', '药物治疗', '检查', '内窥镜检查', '预后状况', '同义词', '发病率', '组织学检查',
 '流行病学', '影像学检查', '相关（症状）', '就诊科室', '遗传因素', '鉴别诊断', '风险评估因素', '社会学', '多发季节', '临床表现', '传播途径', '其他治疗', '部位', '发病年龄', '病理分型', '死亡率', 
 '高危因素', '病因', '放射治疗', '疾病', '辅助治疗', '药物', '阶段', '症状'}
 
所有关系类型( 53 )： {'疾病名称:疾病病史', '疾病名称:疾病阶段', '疾病名称:侵及症状', '药物名称:药物名称', '疾病名称:临床表现', '其他实体:其他实体', '检查名称:检查名称', '疾病名称:发病年龄', 
'疾病名称:高危因素', '社会学名:社会学名', '疾病症状:疾病症状', '疾病名称:病理分型', '疾病名称:预后状况', '疾病名称:多发季节', '疾病名称:预防方法', '疾病名称:转移部位', '疾病名称:风险因素', 
'疾病名称:化疗治疗', '身体部位:身体部位', '疾病名称:发病概率', '疾病名称:治后症状', '疾病名称:实验检查', '疾病名称:外侵部位', '疾病名称:传播途径', '疾病名称:遗传因素', '疾病名称:辅助治疗', 
'疾病名称:相关转化', '疾病名称:疾病病因', '手术治疗:手术治疗', '疾病名称:影像检查', '疾病名称:鉴别诊断', '疾病名称:相关导致', '疾病名称:放射治疗', '疾病名称:性别倾向', '疾病名称:疾病名称', 
'疾病名称:发病部位', '疾病名称:手术治疗', '疾病名称:筛查检查', '疾病名称:多发群体', '其他治疗:其他治疗', '疾病名称:多发地区', '疾病名称:预后生率', '疾病名称:病理生理', '流行病学:流行病学', 
'疾病名称:并发病症', '疾病名称:组织检查', '疾病名称:内镜检查', '疾病名称:就诊科室', '疾病名称:死亡概率', '疾病名称:相关症状', '疾病名称:发病机制', '疾病名称:药物治疗', '疾病名称:辅助检查'}

关系分布统计:  共 100 种关系： Counter({'疾病名称:临床表现': 14603, '疾病名称:药物治疗': 5613, '疾病名称:疾病病因': 3321, '疾病名称:疾病名称': 3189, '疾病名称:并发病症': 2598, '疾病名称:病理分型': 2345, 
'疾病名称:实验检查': 2299, '疾病名称:辅助治疗': 1932, '疾病名称:相关导致': 1811, '疾病名称:影像检查': 1756, '疾病名称:鉴别诊断': 1621, '疾病名称:发病部位': 1480, '疾病名称:高危因素': 1449, '疾病名称:手术治疗': 1119, 
'疾病名称:相关转化': 883, '疾病名称:多发群体': 703, '疾病名称:辅助检查': 692, '疾病名称:风险因素': 620, '疾病名称:发病概率': 507, '疾病名称:相关症状': 503, '疾病名称:预防方法': 478, '疾病名称:组织检查': 388, '疾病名称:发病年龄': 346, 
'药物名称:药物名称': 342, '疾病名称:多发地区': 277, '疾病名称:转移部位': 273, '检查名称:检查名称': 249, '疾病名称:预后状况': 247, '疾病名称:疾病阶段': 242, '疾病名称:内镜检查': 230, '疾病名称:疾病名称:无': 207, '疾病名称:临床表现:无': 190,
 '疾病名称:性别倾向': 183, '疾病名称:化疗治疗': 175, '疾病名称:放射治疗': 172, '疾病名称:疾病病史': 167, '疾病名称:外侵部位': 167, '疾病名称:遗传因素': 159, '社会学名:社会学名': 155, '疾病名称:治后症状': 154, 
 '疾病名称:筛查检查': 131, '疾病名称:病理分型:无': 120, '疾病名称:鉴别诊断:无': 109, '疾病名称:并发病症:无': 105, '疾病名称:预后生率': 97, '疾病名称:多发季节': 94, '疾病名称:死亡概率': 90, '疾病名称:疾病病因:无': 82, 
 '疾病名称:药物治疗:无': 72, '疾病名称:相关导致:无': 68, '疾病名称:发病机制': 66, '其他治疗:其他治疗': 60, '疾病名称:传播途径': 59, '疾病名称:侵及症状': 53, '疾病名称:相关转化:无': 51, '疾病名称:就诊科室': 47, '疾病名称:实验检查:无': 47, 
 '疾病名称:病理生理': 43, '手术治疗:手术治疗': 42, '疾病名称:辅助治疗:无': 39, '疾病名称:影像检查:无': 39, '疾病名称:多发群体:无': 36, '疾病症状:疾病症状': 32, '疾病名称:发病部位:无': 32, '药物名称:药物名称:无': 29, '疾病名称:手术治疗:无': 24, 
 '疾病名称:高危因素:无': 23, '疾病名称:发病概率:无': 22, '疾病名称:相关症状:无': 18, '其他实体:其他实体': 16, '检查名称:检查名称:无': 14, '疾病名称:辅助检查:无': 11, '手术治疗:手术治疗:无': 10, '疾病名称:组织检查:无': 10, '疾病名称:预后状况:无': 10,
  '疾病名称:发病年龄:无': 8, '其他治疗:其他治疗:无': 6, '疾病名称:多发地区:无': 6, '身体部位:身体部位': 5, '疾病名称:疾病病史:无': 5, '社会学名:社会学名:无': 5, '疾病名称:风险因素:无': 5, '疾病名称:预防方法:无': 5, '疾病名称:内镜检查:无': 4,
   '疾病名称:转移部位:无': 4, '流行病学:流行病学': 3, '其他实体:其他实体:无': 3, '疾病名称:外侵部位:无': 3, '疾病名称:预后生率:无': 2, '疾病名称:筛查检查:无': 2, '疾病名称:疾病阶段:无': 2, '疾病名称:治后症状:无': 2, '疾病名称:性别倾向:无': 2,
    '疾病名称:多发季节:无': 2, '身体部位:身体部位:无': 1, '疾病名称:发病机制:无': 1, '疾病名称:侵及症状:无': 1, '疾病名称:遗传因素:无': 1, '疾病症状:疾病症状:无': 1, '疾病名称:传播途径:无': 1})
'''
