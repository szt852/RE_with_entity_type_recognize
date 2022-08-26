import torch
from arguments import get_args
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

def get_optimizer(model, train_dataloader):

    args = get_args()
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    cur_model = model.module if hasattr(model, 'module') else model

    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in cur_model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in cur_model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]  # 筛选模型中哪些参数需要进行权重衰减(也就是L2正则化)，哪些参数不需要进行权重衰减，同时根据筛选条件可知所有bias和LayerNorm相关的参数都不需要进行正则化处理。
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        eps=args.adam_epsilon)   # 定义优化器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, num_training_steps=t_total)  # 动态学习率调整的策略；第31-34行是指定动态学习率的调整策略
    
    embedding_parameters = [
        {'params': [p for p in cur_model.mlp.parameters()]},
        {'params': [p for p in cur_model.extra_token_embeddings.parameters()]}
    ]
    embedding_optimizer = AdamW(
        embedding_parameters, 
        lr=args.learning_rate_for_new_token, 
        eps=args.adam_epsilon)
    embedding_scheduler = get_linear_schedule_with_warmup(
        embedding_optimizer, 
        num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    return optimizer, scheduler, embedding_optimizer, embedding_scheduler
