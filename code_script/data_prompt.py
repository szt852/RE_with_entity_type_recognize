import torch
import numpy as np
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from arguments import get_args

class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""
    def __init__(self, **tensors):
        tensors.values()
        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

    def cuda(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cuda()

    def cpu(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cpu()


class REPromptDataset(DictDataset):

    def __init__(self, path = None, name = None, rel2id = None, tokenizer = None, temps = None, features = None):

        with open(rel2id, "r",encoding='utf-8') as f: # 存放的是关系到关系编号的映射
            self.rel2id = json.loads(f.read())
        if not 'NA' in self.rel2id:
            # self.NA_NUM = self.rel2id['没有关系01'] # 用NA_NUM=13表示no_relationd的编号
            self.NA_NUM = 16 # 用NA_NUM=13表示no_relationd的编号
        # else:
        #     self.NA_NUM = self.rel2id['NA']


        self.num_class = len(self.rel2id) # 使用num_class表示关系的数量
        self.temps = temps
        self.get_labels(tokenizer)

        if features is None:
            self.args = get_args()
            with open(path+"/" + name, "r",encoding='utf8') as f:  # 加载数据集train or test or dev
                features = []
                for line in f.readlines():
                    line = line.rstrip()  # {'token': ['Tom', 'Thabane', 'resigned', 'in', 'October', 'last', 'year', 'to', 'form', 'the', 'All', 'Basotho', 'Convention', '-LRB-', 'ABC', '-RRB-', ',', 'crossing', 'the', 'floor', 'with', '17', 'members', 'of', 'parliament', ',', 'causing', 'constitutional', 'monarch', 'King', 'Letsie', 'III', 'to', 'dissolve', 'parliament', 'and', 'call', 'the', 'snap', 'election', '.'], 'h': {'name': 'All Basotho Convention', 'pos': [10, 13]}, 't': {'name': 'Tom Thabane', 'pos': [0, 2]}, 'relation': 'org:founded_by'}
                    if len(line) > 0:
                        features.append(eval(line))  # features存放文本数据列表token,头实体h,尾实体t,关系relasion: [{'token': ['Tom', 'Thabane', 'resigned', 'in', 'October', 'last', 'year', 'to', 'form', 'the', 'All', 'Basotho', 'Convention', '-LRB-', 'ABC', '-RRB-', ',', 'crossing', 'the', 'floor', 'with', '17', 'members', 'of', 'parliament', ',', 'causing', 'constitutional', 'monarch', 'King', 'Letsie', 'III', 'to', 'dissolve', 'parliament', 'and', 'call', 'the', 'snap', 'election', '.'], 'h': {'name': 'All Basotho Convention', 'pos': [10, 13]}, 't': {'name': 'Tom Thabane', 'pos': [0, 2]}, 'relation': 'org:founded_by'}]
            features = self.list2tensor(features, tokenizer)

        super().__init__(**features)  # super().__init__()，就是继承父类的init方法
    
    def get_labels(self, tokenizer):
        """
        输入：
            tokenizer： 编码器
        过程：
            构建 temp_ids{}:
            构建 set[]:
            构建 prompt_id_2_label[]:
            构建 prompt_label_idx[]:
        输出：null
        """
        total = {}
        self.temp_ids = {}

        for name in self.temps: # 按照每个模板进行循环 name；'per:charges'
            last = 0
            self.temp_ids[name] = {}
            self.temp_ids[name]['label_ids'] = []
            self.temp_ids[name]['mask_ids'] = []

            for index, temp in enumerate(self.temps[name]['temp']):  # 对每个子提示进行循环 index：0 temp:['the', '<mask>']
                _temp = temp.copy() # _temp=['the', '<mask>']
                _labels = self.temps[name]['labels'][index]  #  _labels=person
                _labels_index = [] # 记录label的索引，也就是[mask]的位置索引 _labels_index=[1]

                for i in range(len(_temp)):
                    if _temp[i] == tokenizer.mask_token:
                        _temp[i] = _labels[len(_labels_index)] # _labels只存放label 所以此时_temp的[MASK]变成了person _temp=['the', 'person']
                        _labels_index.append(i)

                # original = tokenizer.encode(" ".join(temp), add_special_tokens=False) # temp=['the', '<mask>'] 的单词编号 original=[627, 50264]
                # final =  tokenizer.encode(" ".join(_temp), add_special_tokens=False) # _temp=['the', 'person'] 的单词编号 final=[627, 621]
                original = tokenizer.encode(temp,add_special_tokens=False)  # temp=['the', '<mask>'] 的单词编号 original=[627, 50264]
                final = tokenizer.encode(_temp,add_special_tokens=False)  # _temp=['the', 'person'] 的单词编号 final=[627, 621]
                print("\n_temp:",_temp)
                print("original----------",original)
                print("final-------------",final)
                # print(tokenizer.encode(["'s"], add_special_tokens=False))
                # print(tokenizer.encode(["'s", 'relative'], add_special_tokens=False))
                # print(tokenizer.encode(["'s", 'relative', 'is'], add_special_tokens=False))
                # original = tokenizer.encode(temp,add_special_tokens=False)  # temp=['the', '<mask>'] 的单词编号 original=[627, 50264]
                # final = tokenizer.encode(_temp,add_special_tokens=False)  # _temp=['the', 'person'] 的单词编号 final=[627, 621]

                # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。断言可以在条件不满足程序运行的情况下直接返回错误。
                assert len(original) == len(final)
                self.temp_ids[name]['label_ids'] += [final[pos] for pos in _labels_index]  # temp_ids[name]['label_ids']中存放标签词label的编号

                for pos in _labels_index:
                    if not last in total:
                        total[last] = {}
                    total[last][final[pos]] = 1 # total={0: {621: 1}} 存放子模板的标签标号和1   -->第二轮 {0: {621: 1}, 1: {7325: 1}, 2: {1340: 1}, 3: {19: 1}} -->第三轮 {0: {621: 1}, 1: {7325: 1}, 2: {1340: 1}, 3: {19: 1}, 4: {515: 1}}
                    last+=1   # last用来存储此时加入total的是一个句子中的第几个标签词
                self.temp_ids[name]['mask_ids'].append(original)  # temp_ids[name]['mask_ids']存放['the', '<mask>'] 的单词编号 original=[627, 50264]

        print (total) # total里存放的是模板中每个位置对应会出现所有label的编号 ，一共有5个类型对应0，1，2，3，4的mask
        """{0: {621: 1, 1651: 1, 10014: 1}, 
            1: {7325: 1, 18: 1, 354: 1}, 
            2: {1340: 1, 962: 1, 4790: 1, 2421: 1, 5221: 1, 5407: 1, 1270: 1, 25385: 1, 2034: 1, 1207: 1, 919: 1, 8850: 1, 334: 1, 3200: 1, 21771: 1, 17117: 1, 4095: 1, 920: 1, 29853: 1, 8540: 1, 26241: 1, 998: 1, 1046: 1, 21821: 1}, 
            3: {19: 1, 15: 1, 11: 1, 9: 1, 30: 1, 16: 1, 21: 1, 34: 1, 7: 1}, 
            4: {515: 1, 1248: 1, 247: 1, 621: 1, 343: 1, 194: 1, 1270: 1, 1651: 1, 6825: 1, 346: 1, 46471: 1, 10014: 1}}"""
        self.set = [(list)((sorted)(set(total[i]))) for i in range(len(total))]
        """set=[[621, 1651, 10014], 
                [18, 354, 7325], 
                [334, 919, 920, 962, 998, 1046, 1207, 1270, 1340, 2034, 2421, 3200, 4095, 4790, 5221, 5407, 8540, 8850, 17117, 21771, 21821, 25385, 26241, 29853], 
                [7, 9, 11, 15, 16, 19, 21, 30, 34],
                [194, 247, 343, 346, 515, 621, 1248, 1270, 1651, 6825, 10014, 46471]]"""
        print ("=================================")
        for i in self.set:
            print (i)
        print ("=================================")

        for name in self.temp_ids: # temp_ids={'per:charges': {'label_ids': [0, 7325, 1340, 19, 515], 'mask_ids': [[627, 50264], [50264, 50264, 50264], [627, 50264]]}, 'per:date_of_death': {'label_ids': [621, 7325, 962, 15, 1248], 'mask_ids': [[627, 50264], [50264, 50264, 50264], [627, 50264]]},
            for j in range(len(self.temp_ids[name]['label_ids'])):
                self.temp_ids[name]['label_ids'][j] = self.set[j].index(
                    self.temp_ids[name]['label_ids'][j])  # 将temp_ids[name]['label_ids']中的标签label编号改为其在set数组中的位置编号 ：'label_ids': [1651, 7325, 5221, 30, 621] --》'label_ids': [1, 2, 14, 7, 5]

        self.prompt_id_2_label = torch.zeros(len(self.temp_ids), len(self.set)).long() # 一个（12，5）的0矩阵
        
        for name in self.temp_ids:
            for j in range(len(self.prompt_id_2_label[self.rel2id[name]])):  # 当name='per:date_of_death'时，self.rel2id[name]=2  len(self.prompt_id_2_label[self.rel2id[name]])一直等于5
                self.prompt_id_2_label[self.rel2id[name]][j] = self.temp_ids[name]['label_ids'][j] # 模板每个位置对应的该位置能出现的label编号 prompt_id_2_label[][]用来按照rel2id里的顺序存放temp_ids[name]['label_ids'],一个temp_ids[name]['label_ids']为一行

        if (get_args().select_device == "cpu"):
            self.prompt_id_2_label = self.prompt_id_2_label.long().cpu()
        else:
            self.prompt_id_2_label = self.prompt_id_2_label.long().cuda()

        self.prompt_label_idx = [torch.Tensor(i).long() for i in self.set]  # prompt_label_idx 是 set的tensor形式
        """"[tensor([621,1651, 10014]), 
            tensor([18,354, 7325]), 
            tensor([334, 919, 920, 962, 998,1046,1207,1270,1340,2034, 2421,3200,4095,4790,5221,5407,8540,8850, 17117, 21771,21821, 25385, 26241, 29853]), 
            tensor([ 7,9, 11, 15, 16, 19, 21, 30, 34]), 
            tensor([194, 247, 343, 346, 515, 621,1248,1270,1651,6825,10014, 46471])]"""

    def save(self, path = None, name = None):
        path = path + "/" + name  + "/"
        np.save(path+"input_ids", self.tensors['input_ids'].numpy())
        np.save(path+"token_type_ids", self.tensors['token_type_ids'].numpy())
        np.save(path+"attention_mask", self.tensors['attention_mask'].numpy())
        np.save(path+"labels", self.tensors['labels'].numpy())
        np.save(path+"mlm_labels", self.tensors['mlm_labels'].numpy())
        np.save(path+"input_flags", self.tensors['input_flags'].numpy())
        # np.save(path+"prompt_label_idx_0", self.prompt_label_idx[0].numpy())
        # np.save(path+"prompt_label_idx_1", self.prompt_label_idx[1].numpy())
        # np.save(path+"prompt_label_idx_2", self.prompt_label_idx[2].numpy())

    @classmethod  # 一般来说，要使用某个类的方法，需要先实例化一个对象再调用方法。而使用@staticmethod或@classmethod，就可以不需要实例化，直接类名.方法名()来调用。这有利于组织代码，把某些应该属于某个类的函数给放到那个类里去，同时有利于命名空间的整洁。
    def load(cls, path = None, name = None, rel2id = None, temps = None, tokenizer = None):
        path = path + "/" + name  + "/"
        features = {}
        features['input_ids'] = torch.Tensor(np.load(path+"input_ids.npy")).long()
        features['token_type_ids'] = torch.Tensor(np.load(path+"token_type_ids.npy")).long()
        features['attention_mask'] = torch.Tensor(np.load(path+"attention_mask.npy")).long()
        features['labels'] = torch.Tensor(np.load(path+"labels.npy")).long()
        features['input_flags'] = torch.Tensor(np.load(path+"input_flags.npy")).long()
        features['mlm_labels'] = torch.Tensor(np.load(path+"mlm_labels.npy")).long()
        res = cls(rel2id = rel2id, features = features, temps = temps, tokenizer = tokenizer)
        # res.prompt_label_idx = [torch.Tensor(np.load(path+"prompt_label_idx_0.npy")).long(),
        #     torch.Tensor(np.load(path+"prompt_label_idx_1.npy")).long(),
        #     torch.Tensor(np.load(path+"prompt_label_idx_2.npy")).long()
        # ]
        return res

    def list2tensor(self, data, tokenizer):
        """输入：
            data:一条数据 [{'token': ['Tom', 'Thabane', 'resigned', ... , '.'], 'h': {'name': 'All Basotho Convention', 'pos': [10, 13]}, 't': {'name': 'Tom Thabane', 'pos': [0, 2]}, 'relation': 'org:founded_by'}]
            tokenizer:模型编码器
        过程：
            对上一步计算出的：input_ids[],token_type_ids[],attention_mask[],input_flags[] 进行补齐
            构建labels：表示关系对应的编号
            构建mlm_labels[]：[MASK]对应位置填1，其余位置填-1
        输出：
            res{}={'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'input_flags': [], 'mlm_labels': [], 'labels': []}
        """
        """'input_ids': [array([  0, 15691,  2032,   873,  1728,  6490,    11,   779,    94,
                                      76,     7,  1026,     5,   404,  7093,  6157,   139,  9127,
                                     111,   574, 26770,    12,  3943,   111, 25733,   387,    12,
                                    2156,  6724,     5,  1929,    19,   601,   453,     9,  3589,
                                    2156,  3735,  6100, 20303,  1745, 40702,   324,  6395,     7,
                                   30887,  3589,     8,   486,     5,  6788,   729,   479,   627,
                                   50264,   404,  7093,  6157,   139,  9127, 50264, 50264, 50264,
                                     627, 50264,  1560,  2032,   873,  1728,     2,     1,     1,
                                       1,     1,     1,     1,     1,     1,     1,     1,     1,
                                       1,     1,     1,     1,     1,     1,     1,     1, ...])]单词编号和 PAD:1
                'token_type_ids': [array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...])]全0 同'input_flags':
                'attention_mask': [array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...])]前1后0
                'mlm_labels': [array([ -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                       -1., -1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1., -1.,  1.,
                                       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,...])] [MASK]1 其余-1
                'labels': [array(41)]"""
        res = {}  # {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'input_flags': [], 'mlm_labels': [], 'labels': []}
        res['input_ids'] = []
        res['token_type_ids'] = []
        res['attention_mask'] = []
        res['input_flags'] = []
        res['mlm_labels'] = []
        res['labels'] = []

        for index, i in enumerate(tqdm(data)):
            input_ids, token_type_ids, input_flags = self.tokenize(i, tokenizer)  # 根据句子的头尾实体组装prompt模板
            attention_mask = [1] * len(input_ids)  # attention_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            padding_length = self.args.max_seq_length - len(input_ids)  # padding_length=512-70=442

            if padding_length > 0:  # 对 input_ids的 PAD进行填充，其中tokenizer.pad_token_id=1
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)  # attention_mask=[1*70,0*442]
                token_type_ids = token_type_ids + ([0] * padding_length)  # token_type_ids,input_flags=[0*512]
                input_flags = input_flags + ([0] * padding_length)

            assert len(input_ids) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(token_type_ids) == self.args.max_seq_length
            assert len(input_flags) == self.args.max_seq_length

            label = self.rel2id[i['relation']]  # label表示关系对应的编号 rel2id['org:founded_by']=41
            res['input_ids'].append(np.array(input_ids))
            res['token_type_ids'].append(np.array(token_type_ids))
            res['attention_mask'].append(np.array(attention_mask))
            res['input_flags'].append(np.array(input_flags))
            res['labels'].append(np.array(label))
            mask_pos = np.where(res['input_ids'][-1] == tokenizer.mask_token_id)[
                0]  # res['input_ids']中编号为[MASK]的编号50264的位置
            mlm_labels = np.ones(self.args.max_seq_length) * (-1)  # mlm_labels=[-1*512]
            mlm_labels[mask_pos] = 1  # 将mlm_labels中的mask_pos位置设置为1
            res['mlm_labels'].append(mlm_labels)
        for key in res:
            res[key] = np.array(res[key])
            res[key] = torch.Tensor(res[key]).long()
        return res

    # 组装prompt
    def tokenize(self, item, tokenizer): # {'token': ['Tom', 'Thabane', 'resigned', 'in', 'October', 'last', 'year', 'to', 'form', 'the', 'All', 'Basotho', 'Convention', '-LRB-', 'ABC', '-RRB-', ',', 'crossing', 'the', 'floor', 'with', '17', 'members', 'of', 'parliament', ',', 'causing', 'constitutional', 'monarch', 'King', 'Letsie', 'III', 'to', 'dissolve', 'parliament', 'and', 'call', 'the', 'snap', 'election', '.'], 'h': {'name': 'All Basotho Convention', 'pos': [10, 13]}, 't': {'name': 'Tom Thabane', 'pos': [0, 2]}, 'relation': 'org:founded_by'}
        """
        输入：
            item： 文本数据，形式为{'token': [], 'h': {'name': '', 'pos': []}, 't': {'name': '', 'pos': []}, 'relation': 'org:founded_by'}
            tokenizer： 编码器
        过程：
            组装prompt[]: prompt=temp_ids[rel_name]['mask_ids'][0] + e1 + self.temp_ids[rel_name]['mask_ids'][1] + self.temp_ids[rel_name]['mask_ids'][2] + e2
                temp_ids[rel_name]['mask_ids'][0]     = [627, 50264]
                e1：头实体编码                           = [404, 7093, 6157, 139, 9127]
                self.temp_ids[rel_name]['mask_ids'][1]= [50264, 50264, 50264]
                self.temp_ids[rel_name]['mask_ids'][2]= [627, 50264]
                e2：尾实体编码                           = [1560, 2032, 873, 1728]
                prompt='the [MASK]' + 'token_id(e1)' + '[MASK] [MASK] [MASK]' + 'the [MASK]' + 'token_id(e2)'
                prompt=[627, 50264]+[404, 7093, 6157, 139, 9127]+[50264, 50264, 50264]+[627, 50264]+[1560, 2032, 873, 1728]
        输出：
            input_ids[]：使用原始句子和模板组装：input_ids=[cls]+sentence+prompt+[sep]，使用单词对应索引表示
            token_type_ids[]：[0*70]
            input_flags[]：[0*70]
        """
        sentence = item['token']  # sentence=['Tom', 'Thabane', 'resigned', 'in', 'October', 'last', 'year', 'to', 'form', 'the', 'All', 'Basotho', 'Convention', '-LRB-', 'ABC', '-RRB-', ',', 'crossing', 'the', 'floor', 'with', '17', 'members', 'of', 'parliament', ',', 'causing', 'constitutional', 'monarch', 'King', 'Letsie', 'III', 'to', 'dissolve', 'parliament', 'and', 'call', 'the', 'snap', 'election', '.']
        pos_head = item['h']  # {'name': 'All Basotho Convention', 'pos': [10, 13]}
        pos_tail = item['t']  # {'name': 'Tom Thabane', 'pos': [0, 2]}
        rel_name = item['relation']  # 'org:founded_by'

        temp = self.temps[rel_name]  # 'org:founded_by' 对应的模板 {'name': 'org:founded_by', 'temp': [['the', '<mask>'], ['<mask>', '<mask>', '<mask>'], ['the', '<mask>']], 'labels': [('organization',), ('was', 'founded', 'by'), ('person',)]}

        sentence = " ".join(sentence) # 将句子拼起来 sentence='Tom Thabane resigned in October last year to form the All Basotho Convention -LRB- ABC -RRB- , crossing the floor with 17 members of parliament , causing constitutional monarch King Letsie III to dissolve parliament and call the snap election .'
        sentence = tokenizer.encode(sentence, add_special_tokens=False)  # sentence中的单词转变成单词编码 长度变为52？？？？根据使用模型的不同其分词会有区别 cuting--》'cut', '##ing' sentence=[15691, 2032, 873, 1728, 6490, 11, 779, 94, 76, 7, 1026, 5, 404, 7093, 6157, 139, 9127, 111, 574, 26770, 12, 3943, 111, 25733, 387, 12, 2156, 6724, 5, 1929, 19, 601, 453, 9, 3589, 2156, 3735, 6100, 20303, 1745, 40702, 324, 6395, 7, 30887, 3589, 8, 486, 5, 6788, 729, 479]
        # e1 = tokenizer.encode( pos_head['name'], add_special_tokens=False)[1:]  # 'was All Basotho Convention'的编码  e1：头实体编码 =[404, 7093, 6157, 139, 9127]
        # e2 = tokenizer.encode( pos_tail['name']+"的", add_special_tokens=False)[1:]  # 'was Tom Thabane'的编码  e2：尾实体编码 =[1560, 2032, 873, 1728]
        # prompt = e1 + self.temp_ids[rel_name]['mask_ids'][0] + e2 + self.temp_ids[rel_name]['mask_ids'][1]    # rel_name = 'org:founded_by'

        if (get_args().data_dir == "../datasets/diakg_temp1" or get_args().data_dir == "../datasets/CMeIE_temp1"):
            # β细胞胰岛素分泌功能已开始逐渐下降(2b级)[MASK][MASK][MASK][MASK][MASK]糖尿病的[MASK][MASK][MASK][MASK]。
            e1 = tokenizer.encode(pos_head['name'], add_special_tokens=False)  # 'was All Basotho Convention'的编码  e1：头实体编码 =[404, 7093, 6157, 139, 9127]
            e2 = tokenizer.encode(pos_tail['name']+"的", add_special_tokens=False)  # 'was Tom Thabane'的编码  e2：尾实体编码 =[1560, 2032, 873, 1728]
            prompt = e1 + self.temp_ids[rel_name]['mask_ids'][0] + e2 + self.temp_ids[rel_name]['mask_ids'][1]  # rel_name = 'org:founded_by'
        # elif (get_args().data_dir == "../datasets/diakg_temp2"):
        #     # β细胞胰岛素分泌功能已开始逐渐下降(2b级)是[MASK][MASK][MASK][MASK]糖尿病的[MASK][MASK][MASK][MASK]。
        #     e1 = tokenizer.encode(pos_head['name'] + "的",add_special_tokens=False)  # 'was All Basotho Convention'的编码  e1：头实体编码 =[404, 7093, 6157, 139, 9127]
        #     e2 = tokenizer.encode(pos_tail['name'] + "是",add_special_tokens=False)  # 'was Tom Thabane'的编码  e2：尾实体编码 =[1560, 2032, 873, 1728]
        #     prompt = e2 + self.temp_ids[rel_name]['mask_ids'][0] + e1 + self.temp_ids[rel_name]['mask_ids'][1]  # rel_name = 'org:founded_by'
        # elif (get_args().data_dir == "../datasets/CMeIE"):
        #     # 糖尿病是[MASK][MASK][MASK][MASK]。β细胞胰岛素分泌功能已开始逐渐下降(2b级)是[MASK][MASK][MASK][MASK]。
        #     e1 = tokenizer.encode(pos_head['name'] + "是",add_special_tokens=False)  # 'was All Basotho Convention'的编码  e1：头实体编码 =[404, 7093, 6157, 139, 9127]
        #     e2 = tokenizer.encode("。" + pos_tail['name'] + "是",add_special_tokens=False)  # 'was Tom Thabane'的编码  e2：尾实体编码 =[1560, 2032, 873, 1728]
        #     prompt = e1 + self.temp_ids[rel_name]['mask_ids'][0] + e2 + self.temp_ids[rel_name]['mask_ids'][1]  # rel_name = 'org:founded_by'
        elif (get_args().data_dir == "../datasets/CMeIE_temp2" or get_args().data_dir == "../datasets/diakg_temp2"):
            # 糖尿病是[MASK][MASK][MASK][MASK]。β细胞胰岛素分泌功能已开始逐渐下降(2b级)是[MASK][MASK][MASK][MASK]，二者[MASK]关系。
            e1 = tokenizer.encode(pos_head['name'] + "是",add_special_tokens=False)  # 'was All Basotho Convention'的编码  e1：头实体编码 =[404, 7093, 6157, 139, 9127]
            e2 = tokenizer.encode("," + pos_tail['name'] + "是",add_special_tokens=False)  # 'was Tom Thabane'的编码  e2：尾实体编码 =[1560, 2032, 873, 1728]
            prompt = e1 + self.temp_ids[rel_name]['mask_ids'][0] + e2 + self.temp_ids[rel_name]['mask_ids'][1] \
                     + self.temp_ids[rel_name]['mask_ids'][2] # rel_name = 'org:founded_by'
        """
        temp_ids['org:founded_by']: {'label_ids': [1, 2, 13, 7, 5], 'mask_ids': [[627, 50264], [50264, 50264, 50264], [627, 50264]]}
        temp_ids[rel_name]['mask_ids'][0]     = [627, 50264]
        e1：头实体编码                           = [404, 7093, 6157, 139, 9127]
        self.temp_ids[rel_name]['mask_ids'][1]= [50264, 50264, 50264]
        self.temp_ids[rel_name]['mask_ids'][2]= [627, 50264]
        e2：尾实体编码                           = [1560, 2032, 873, 1728]
        将头尾实体放入模板：['the', '<mask>'],['was All Basotho Convention'] ['<mask>', '<mask>', '<mask>'], ['the', '<mask>']['was Tom Thabane']
        prompt组装后：[627, 50264, 404, 7093, 6157, 139, 9127, 50264, 50264, 50264, 627, 50264, 1560, 2032, 873, 1728]
        ？？？这里暂时还没有把关系标签组装进去？？？
        """
        #  + \
        #  [tokenizer.unk_token_id, tokenizer.unk_token_id]

        flags = []   # 有单词的地方记录为0 flags =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # last = 0
        # for i in prompt:
            # if i == tokenizer.unk_token_id:
            #     last+=1
            #     flags.append(last)
            # else:
            # flags.append(0)
        
        tokens = sentence + prompt  # tokens：52+16 句子 和 模板 拼接起来[15691, 2032, 873, 1728, 6490, 11, 779, 94, 76, 7, 1026, 5, 404, 7093, 6157, 139, 9127, 111, 574, 26770, 12, 3943, 111, 25733, 387, 12, 2156, 6724, 5, 1929, 19, 601, 453, 9, 3589, 2156, 3735, 6100, 20303, 1745, 40702, 324, 6395, 7, 30887, 3589, 8, 486, 5, 6788, 729, 479, 627, 50264, 404, 7093, 6157, 139, 9127, 50264, 50264, 50264, 627, 50264, 1560, 2032, 873, 1728]
        flags = [0 for i in range(len(tokens))]  # flags：[0*68] 将tokens里有单词编码的位置设为0
        # tokens = prompt + sentence
        # flags =  flags + [0 for i in range(len(sentence))]        
        
        tokens = self.truncate(tokens, max_length = self.args.max_seq_length - tokenizer.num_special_tokens_to_add(False))  # 对tokens长度进行裁剪
        flags = self.truncate(flags, max_length = self.args.max_seq_length - tokenizer.num_special_tokens_to_add(False))  # 对flags长度进行裁剪

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens)  # build_inputs_with_special_tokens()函数：如果只有一句话就在首尾分别加上cls_token_id（0），sep_token_id（2）。  input_ids=[0, 15691, 2032, 873, 1728, 6490, 11, 779, 94, 76, 7, 1026, 5, 404, 7093, 6157, 139, 9127, 111, 574, 26770, 12, 3943, 111, 25733, 387, 12, 2156, 6724, 5, 1929, 19, 601, 453, 9, 3589, 2156, 3735, 6100, 20303, 1745, 40702, 324, 6395, 7, 30887, 3589, 8, 486, 5, 6788, 729, 479, 627, 50264, 404, 7093, 6157, 139, 9127, 50264, 50264, 50264, 627, 50264, 1560, 2032, 873, 1728, 2]
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens)  # create_token_type_ids_from_sequences()函数： 如果只有一句话返回列表 len(cls + token_ids_0 + sep) * [0]。   token_type_ids =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        input_flags = tokenizer.build_inputs_with_special_tokens(flags)  # input_flags=[0, 0*68,  2]
        input_flags[0] = 0
        input_flags[-1] = 0  #  input_flags=[0*70]
        assert len(input_ids) == len(input_flags)
        assert len(input_ids) == len(token_type_ids)  # input_ids, token_type_ids, input_flags三者长度对齐
        return input_ids, token_type_ids, input_flags

    """对句子长度进行裁剪"""
    def truncate(self, seq, max_length):  # max_length=510
        if len(seq) <= max_length:
            return seq
        else:
            print ("=========")
            return seq[len(seq) - max_length:] # 如果长度大于510就取前510个
