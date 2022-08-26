import torch
import torch.nn as nn
from arguments import get_model_classes, get_args

class Model(torch.nn.Module):

    def __init__(self, args, tokenizer = None, prompt_label_idx = None):
        super().__init__()
        model_classes = get_model_classes()   # model_classes存储的是一个模型字典
        model_config = model_classes[args.model_type]  # 选择模型，加载模型配置 model_type=roberta ：{'config': <class 'transformers.models.roberta.configuration_roberta.RobertaConfig'>, 'tokenizer': <class 'transformers.models.roberta.tokenization_roberta.RobertaTokenizer'>, 'model': <class 'transformers.models.roberta.modeling_roberta.RobertaModel'>}

        self.prompt_label_idx = prompt_label_idx  # 每个位置对应的所有可能出现的label的编号

        self.model = model_config['model'].from_pretrained(
            args.model_name_or_path,
            return_dict=False,
            cache_dir=args.cache_dir if args.cache_dir else None)  # 加载预训练模型

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.ReLU(),
            # nn.Dropout(p=args.dropout_prob),
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            )

        self.extra_token_embeddings = nn.Embedding(args.new_tokens, self.model.config.hidden_size) # new_tokens=5

    def forward(self, input_ids, attention_mask, token_type_ids, input_flags, mlm_labels, labels):  # input_ids = {Tensor: (8,512)} = attention_mask, token_type_ids, input_flags, mlm_labels
        raw_embeddings = self.model.embeddings.word_embeddings(input_ids)  # roberta原始的词嵌入 raw_embeddings = {Tensor: (8,512,1024)}
        new_token_embeddings = self.mlp(self.extra_token_embeddings.weight)  # 新的嵌入？？？ new_token_embeddings = {Tensor: (5,1024)}
        new_embeddings = new_token_embeddings[input_flags]  # 这里相当于按照新的嵌入吧input_flags扩大一维 new_embeddings = {Tensor: (8,512,1024)} 最有一个维度是一样的，应该是初始化操作
        inputs_embeds = torch.where(input_flags.unsqueeze(-1) > 0, new_embeddings,raw_embeddings)  # torch.where（）函数  input_flags.unsqueeze(-1)的张量中大于0的用 new_embeddings对应元素填充，其他位置用raw_embeddings中对应位置元素填充 ，一开始都是0，那么就是直接用原始嵌入在填充？？？
        hidden_states, _ = self.model(inputs_embeds=inputs_embeds,attention_mask=attention_mask, token_type_ids=token_type_ids)  # hidden_states = {Tensor:(8,512,1024))}
        hidden_states = hidden_states[mlm_labels >= 0].view(input_ids.size(0), len(self.prompt_label_idx),-1)  # 取hidden_states中[MASK]的位置的编码,并将维度变为.view(8，5，-1）--> hidden_states={Tensor: (8,5,1024)}
        logits = [
            torch.mm(hidden_states[:,index,:],  self.model.embeddings.word_embeddings.weight[i].transpose(1,0))
            for index, i in enumerate(self.prompt_label_idx)
        ]   # # torch.mm()矩阵乘法  prompt_label_idx存储的是[MASK]位置的label的编号 ①index=0,i=[621,1651,10014] ②index=1,i=[18,354,7325]...⑤   hidden_states[:,index,:]=(8,1024)    预训练模型的嵌入表：model.embeddings.word_embeddings.weight=(50265,1024)   model.embeddings.word_embeddings.weight[i]=(3,1024)     model.embeddings.word_embeddings.weight[i].transpose(1,0)=(1024,3)    (8,1024)*(1024,3)=(8,3)  所以最终得到一个（8，3）的向量
        return logits

def get_model(tokenizer, prompt_label_idx,device):
    args = get_args()
    model = Model(args, tokenizer, prompt_label_idx)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if(device=="cpu"):
        model.cpu()
    else:
        model.cuda()
    return model

def get_tokenizer(special=[]):
    """选择预训练模型"""
    args = get_args()
    model_classes = get_model_classes()
    # 选择模型
    model_config = model_classes[args.model_type]
    # 选择token
    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.add_tokens(special)
    return tokenizer