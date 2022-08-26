from arguments import get_args_parser

def get_temps(tokenizer):
    args = get_args_parser()#用户设定的参数
    temps = {}#定义一个字典temps，存放所有的[MASK]模板和模板中[MASK]位置的答案label
    # 加载模板文件
    with open(args.data_dir + "/" + args.temps, "r",encoding='utf-8') as f:
        if (args.data_dir=="../datasets/diakg_temp1" or args.data_dir == "../datasets/CMeIE_temp1"):
            for i in f.readlines():
                i = i.strip().split(",") #0 per:charges	person	was	charged	with	event --》['0', 'per:charges', 'person', 'was', 'charged', 'with', 'event']
                info = {}
                info['name'] = i[0].strip()
                info['temp'] = [
                        [tokenizer.mask_token,tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
                        [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token]
                ]
                print (i)
                info['labels'] = [
                    (i[1],i[2],i[3],i[4],i[5]),
                    (i[6],i[7],i[8],i[9]),
                ]
                print(info) #一个info里面有三类信息关系名name,模板temp,模板答案labels info={'name': 'per:charges', 'temp': [['the', '<mask>'], ['<mask>', '<mask>', '<mask>'], ['the', '<mask>']], 'labels': [('person',), ('was', 'charged', 'with'), ('event',)]}
                temps[info['name']] = info #将info加入temps字典中。 共42个字典项
        # elif (args.data_dir=="../datasets/diakg_temp2"):
        #     for i in f.readlines():
        #         i = i.strip().split(",") #0 per:charges	person	was	charged	with	event --》['0', 'per:charges', 'person', 'was', 'charged', 'with', 'event']
        #         info = {}
        #         info['name'] = i[0].strip()
        #         info['temp'] = [
        #             [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
        #             [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
        #             ["，", "二", "者", tokenizer.mask_token, "关", "系"]
        #         ]
        #         print (i)
        #         info['labels'] = [
        #             (i[1], i[2], i[3], i[4]),
        #             (i[5], i[6], i[7], i[8]),
        #             (i[9])
        #         ]
        #         print(info) #一个info里面有三类信息关系名name,模板temp,模板答案labels info={'name': 'per:charges', 'temp': [['the', '<mask>'], ['<mask>', '<mask>', '<mask>'], ['the', '<mask>']], 'labels': [('person',), ('was', 'charged', 'with'), ('event',)]}
        #         temps[info['name']] = info #将info加入temps字典中。 共42个字典项
        # elif (args.data_dir == "../datasets/CMeIE_temp1"):
        #     for i in f.readlines():
        #         i = i.strip().split(",") #0 per:charges	person	was	charged	with	event --》['0', 'per:charges', 'person', 'was', 'charged', 'with', 'event']
        #         info = {}
        #         info['name'] = i[0].strip()
        #         info['temp'] = [
        #                 [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
        #                 [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
        #         ]
        #         print (i)
        #         info['labels'] = [
        #             (i[1],i[2],i[3],i[4]),
        #             (i[5],i[6],i[7],i[8]),
        #         ]
        #         print(info) #一个info里面有三类信息关系名name,模板temp,模板答案labels info={'name': 'per:charges', 'temp': [['the', '<mask>'], ['<mask>', '<mask>', '<mask>'], ['the', '<mask>']], 'labels': [('person',), ('was', 'charged', 'with'), ('event',)]}
        #         temps[info['name']] = info #将info加入temps字典中。 共42个字典项
        elif (args.data_dir == "../datasets/CMeIE_temp2" or args.data_dir=="../datasets/diakg_temp2"):
            for i in f.readlines():
                i = i.strip().split(",") #0 per:charges	person	was	charged	with	event --》['0', 'per:charges', 'person', 'was', 'charged', 'with', 'event']
                info = {}
                info['name'] = i[0].strip()
                info['temp'] = [
                        [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
                        [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
                        ["，","二","者",tokenizer.mask_token,"关","系"]
                ]
                print (i)
                info['labels'] = [
                    (i[1],i[2],i[3],i[4]),
                    (i[5],i[6],i[7],i[8]),
                    (i[9])
                ]
                print(info) #一个info里面有三类信息关系名name,模板temp,模板答案labels info={'name': 'per:charges', 'temp': [['the', '<mask>'], ['<mask>', '<mask>', '<mask>'], ['the', '<mask>']], 'labels': [('person',), ('was', 'charged', 'with'), ('event',)]}
                temps[info['name']] = info #将info加入temps字典中。 共42个字典项
    return temps
