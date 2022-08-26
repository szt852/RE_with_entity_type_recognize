'''
---输入数据文件路径: ../datasets/diakg/diakg_no_re.txt
---输入数据量： 160142
---输出数据量： 50242
---输入数据文件路径: ../datasets/diakg/diakg_have_re.txt
---输入数据量： 8643
---输出数据量： 8643
'''

def del_abs_no_re(diakg_no_re_path,diakg_no_re_ok_path):
    '''根据医疗知识和实体之间的关系结构，找到明显不会有关系的数据,将其剔除,按照实体关系结构图删掉一些明显不存在的关系'''
    diakg_no_re = []
    diakg_no_re_ok = []
    with open(diakg_no_re_path,'r',encoding='utf-8') as f:
        diakg_no_re = f.readlines()
    # print(diakg_no_re)
    f.close()
    print("---输入数据文件路径:",diakg_no_re_path)
    print("---输入数据量：",len(diakg_no_re))

    for i in diakg_no_re:
        i = eval(i)
        # print(i['h']['type'])
        if (i['h']['type'] == '疾病名称' and i['t']['type'] == '不良反应'):
            if(i['relation']=='没有关系'):
                i['relation'] = '没有关系01'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '疾病名称' and i['t']['type'] == '检查方法'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系02'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '疾病名称' and i['t']['type'] == '临床表现'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系03'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '疾病名称' and i['t']['type'] == '非药治疗'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系04'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '疾病名称' and i['t']['type'] == '药品名称'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系05'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '疾病名称' and i['t']['type'] == '患病部位'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系06'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '疾病名称' and i['t']['type'] == '患病病因'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系07'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '疾病名称' and i['t']['type'] == '发病机制'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系08'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '疾病名称' and i['t']['type'] == '临床手术'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系09'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '疾病名称' and i['t']['type'] == '分期分型'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系10'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '疾病名称' and i['t']['type'] == '检查指标'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系11'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '药品名称' and i['t']['type'] == '用药频率'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系12'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '药品名称' and i['t']['type'] == '持续时间'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系13'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '药品名称' and i['t']['type'] == '用药剂量'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系14'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '药品名称' and i['t']['type'] == '用药方法'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系15'
            diakg_no_re_ok.append(i)
        elif (i['h']['type'] == '药品名称' and i['t']['type'] == '不良反应'):
            if (i['relation'] == '没有关系'):
                i['relation'] = '没有关系16'
            diakg_no_re_ok.append(i)
        # else:
        #     print("---i:",i)

    # print(diakg_no_re_ok)
    print("---输出数据量：",len(diakg_no_re_ok))

    with open(diakg_no_re_ok_path,'w+',encoding='utf-8') as f:
        for i in diakg_no_re_ok:
            f.write(str(i)+"\n")
    f.close()

# 对没有关系的数据进行初步关系判断
diakg_no_re_path = '../datasets/diakg/diakg_no_re.txt'
diakg_no_re_ok_path = '../datasets/diakg/diakg_no_re_ok.txt'
del_abs_no_re(diakg_no_re_path,diakg_no_re_ok_path)

# 对有关系的数据进行一个检查
diakg_have_re_path = '../datasets/diakg/diakg_have_re.txt'
diakg_have_re_ok_path = '../datasets/diakg/diakg_have_re_ok.txt'
del_abs_no_re(diakg_have_re_path,diakg_have_re_ok_path)