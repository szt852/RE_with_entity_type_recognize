from arguments import get_args_parser
from templating import get_temps
from modeling import get_model, get_tokenizer
from data_prompt import REPromptDataset
from optimizing import get_optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
import random
import os
import datetime


def f1_score(output, label, rel_num, na_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()
    # no_re_label_id = list(range(16,32))
    for i in range(len(output)):
        guess = output[i]  # 模型对第i个句子预测值
        gold = label[i]  # 第i个句子的真实值

        if (args.data_dir == '../datasets/diakg_temp1' or args.data_dir == '../datasets/diakg_temp2'):
            if (guess > 15):
                guess_by_relation[16] += 1
            if (guess <= 15):
                guess_by_relation[guess] += 1
            if (gold > 15):
                gold_by_relation[16] += 1
            if (gold <= 15):
                gold_by_relation[gold] += 1
            if gold == guess:
                if (gold > 15):
                    correct_by_relation[16] += 1
                if (gold <= 15):
                    correct_by_relation[gold] += 1
            rel_num = 17
        elif (args.data_dir == '../datasets/CMeIE_temp1' or args.data_dir == '../datasets/CMeIE_temp2'):
            if (guess > 52):
                guess_by_relation[53] += 1
            if (guess <= 52):
                guess_by_relation[guess] += 1
            if (gold > 52):
                gold_by_relation[53] += 1
            if (gold <= 52):
                gold_by_relation[gold] += 1
            if gold == guess:
                if (gold > 52):
                    correct_by_relation[53] += 1
                if (gold <= 52):
                    correct_by_relation[gold] += 1
            rel_num = 54

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()

    for i in range(0, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]  # 计算recall
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]  # 计算precision
        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    # prec = 0
    # recall = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        # recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        recall = sum(recall_by_relation.values()) / len(recall_by_relation)
        # prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        prec = sum(prec_by_relation.values()) / len(prec_by_relation)
        micro_f1 = 2 * recall * prec / (recall + prec)

    return micro_f1, f1_by_relation, prec, recall


def evaluate(model, dataset, dataloader):
    model.eval()
    scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits = model(**batch)
            res = []
            for i in dataset.prompt_id_2_label:
                _res = 0.0
                for j in range(len(i)):
                    _res += logits[j][:, i[j]]
                _res = _res.detach().cpu()  # 神经网络的训练有时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整。或者训练部分分支网络，并不让其梯度对主网络的梯度造成影响.这时候我们就需要使用detach()函数来切断一些分支的反向传播.
                res.append(_res)
            logits = torch.stack(res, 0).transpose(1, 0)
            labels = batch['labels'].detach().cpu().tolist()
            all_labels += labels
            scores.append(logits.cpu().detach())
        scores = torch.cat(scores, 0)
        scores = scores.detach().cpu().numpy()
        all_labels = np.array(all_labels)
        np.save("scores.npy", scores)
        np.save("all_labels.npy", all_labels)

        pred = np.argmax(scores, axis=-1)
        mi_f1, ma_f1, prec, recall = f1_score(pred, all_labels, dataset.num_class, dataset.NA_NUM)
        return mi_f1, ma_f1, prec, recall


args = get_args_parser()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if (args.select_device == "cpu"):
        torch.manual_seed(seed)  # 为CPU中设置种子，生成随机数：
    elif args.n_gpu == 1:
        torch.cuda.manual_seed(seed)  # 特定GPU设置种子，生成随机数：
    elif args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。

# 创建保存模型的目录
re_model_name = args.model_name_or_path
re_model_name = re_model_name.replace('/' , '-')
if not os.path.exists(args.output_dir + "/" + re_model_name):
    os.mkdir(args.output_dir + "/" + re_model_name)

set_seed(args.seed)
# 选择模型和token
tokenizer = get_tokenizer(special=[])
# 构建模板字典temps
temps = get_temps(tokenizer)

# 数据处理及保存
# data_make_save()
dataset = REPromptDataset(
    path=args.data_dir,
    name='train.txt',
    rel2id=args.data_dir + "/" + "rel2id.json",
    temps=temps,
    tokenizer=tokenizer, )
dataset.save(path=args.output_dir, name="train")

dataset = REPromptDataset(
    path=args.data_dir,
    name='dev.txt',
    rel2id=args.data_dir + "/" + "rel2id.json",
    temps=temps,
    tokenizer=tokenizer)
dataset.save(path=args.output_dir, name="dev")

dataset = REPromptDataset(
    path=args.data_dir,
    name='test.txt',
    rel2id=args.data_dir + "/" + "rel2id.json",
    temps=temps,
    tokenizer=tokenizer)
dataset.save(path=args.output_dir, name="test")

# 数据加载
train_dataset = REPromptDataset.load(
    path=args.output_dir,
    name="train",
    temps=temps,
    tokenizer=tokenizer,
    rel2id=args.data_dir + "/" + "rel2id.json")

val_dataset = REPromptDataset.load(
    path=args.output_dir,
    name="dev",
    temps=temps,
    tokenizer=tokenizer,
    rel2id=args.data_dir + "/" + "rel2id.json")

test_dataset = REPromptDataset.load(
    path=args.output_dir,
    name="test",
    temps=temps,
    tokenizer=tokenizer,
    rel2id=args.data_dir + "/" + "rel2id.json")

device = torch.device(args.select_device)
train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

# train_dataset.to(device)
if (args.select_device == "cpu"):
    train_dataset.cpu()
    val_dataset.cpu()
    test_dataset.cpu()
else:
    train_dataset.cuda()
    val_dataset.cuda()
    test_dataset.cuda()

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_batch_size // 2)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=train_batch_size // 2)

model = get_model(tokenizer, train_dataset.prompt_label_idx, args.select_device)
optimizer, scheduler, optimizer_new_token, scheduler_new_token = get_optimizer(model, train_dataloader)
criterion = nn.CrossEntropyLoss()

mx_res = 0.0
hist_mi_f1 = []
hist_ma_f1 = []
mx_epoch = None
last_epoch = None


running_f1_list = []    # 保存每个epoch的评估效果
# 开始时间
start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("start_time: " + str(start_time))

for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
    model.train()
    model.zero_grad()
    tr_loss = 0.0
    global_step = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        logits = model(**batch)
        labels = train_dataset.prompt_id_2_label[batch['labels']]  # labels表示关系对应的编号  batch['labels']=[13, 30, 13, 11, 13, 13, 13, 13] --》 根据[13, 30, 13, 11, 13, 13, 13, 13]里的关系对应编号选择相应的prompt模板：prompt_id_2_label[[13, 30, 13, 11, 13, 13, 13, 13]]

        loss = 0.0
        for index, i in enumerate(logits):
            loss += criterion(i, labels[:,index])  # [MASK]位置预测的损失： 计算8个句子中每个句子的[MASK]位置的logits和labels的交叉熵loss  对于第1个位置,一共有三种可能的取值 计算出的logits i=(8,3)   取第一个位置的标签（labels的第一列）计算loss ：labels[:,index]=labels[:,0]=tensor([2,0,2,0,2,2,2,2])
        loss /= len(logits)

        res = []
        for i in train_dataset.prompt_id_2_label:
            _res = 0.0
            for j in range(len(i)):  # j=0~4
                _res += logits[j][:, i[j]]  # j=0时 logits[j][:, i[j]]=logits[0][:, i[0]]
            res.append(_res)  # 按照i=[1,2,13,2,6]在logits中选择每个位置对应标签的logits列向量(8,1),将五个位置的logits列向量进行求和操作
        final_logits = torch.stack(res, 0).transpose(1,0)  # 各个模板logits和组成的二维矩阵的转置，句子对应模板的概率：torch.stack扩充维度？？？  将res (42,8)进行转置-->final_logits (8,42)
        loss += criterion(final_logits, batch['labels'])  # loss = [MASK]位置预测的损失 + 模板logits的损失：

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()  # loss.backward()函数的作用是根据loss来计算网络参数的梯度
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # clip_grad_norm_()梯度裁剪
            optimizer.step()  # 优化器的作用就是针对计算得到的参数梯度对网络参数进行更新，需要两个东西：1.当前的网络模型的参数空间2.反向传播的梯度信息（即backward计算得到的信息）
            scheduler.step()
            optimizer_new_token.step()
            scheduler_new_token.step()
            model.zero_grad()
            print(args)
            global_step += 1
            # print (tr_loss/global_step, mx_res)
            print("tr_loss/global_step:", tr_loss / global_step, " mx_res:", mx_res, " current_epoch:", epoch)

    mi_f1, ma_f1, prec, recall = evaluate(model, val_dataset, val_dataloader)
    hist_mi_f1.append(mi_f1)
    hist_ma_f1.append(ma_f1)
    if mi_f1 > mx_res:
        mx_res = mi_f1
        mx_epoch = epoch
        torch.save(model.state_dict(), args.output_dir + "/" + re_model_name + "/" + 'parameter' + '_max' + ".pkl")
    torch.save(model.state_dict(), args.output_dir + "/" + re_model_name + "/" + 'parameter' + str(epoch) + ".pkl")
    last_epoch = epoch
    print("epoch:", epoch, "  mi_f1：", mi_f1, "  prec:", prec, "  recall:", recall, "  ma_f1:", ma_f1)
    running_f1_list.append("epoch:" + str(epoch)+ "  mi_f1：" + str(mi_f1) + "  prec:" + str(prec)+ "  recall:" + str(recall) + "  ma_f1:" + str(ma_f1))


model.load_state_dict(torch.load(args.output_dir + "/" + re_model_name + "/" + 'parameter' + str(last_epoch) + ".pkl"))
last_epoch_mi_f1, last_epoch_ma_f1, last_epoch_prec, last_epoch_recall = evaluate(model, test_dataset, test_dataloader)
print("last_epoch:", last_epoch, "  mi_f1：", last_epoch_mi_f1, "  prec:", last_epoch_prec, "  recall:", last_epoch_recall, "  ma_f1:", last_epoch_ma_f1)

model.load_state_dict(torch.load(args.output_dir + "/" + re_model_name + "/" + 'parameter' + '_max'  + ".pkl"))
mi_f1, ma_f1, prec, recall = evaluate(model, test_dataset, test_dataloader)
print("mx_epoch:", mx_epoch, "  mi_f1：", mi_f1, "  prec:", prec, "  recall:", recall, "  ma_f1:", ma_f1)


complate_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("complate_time: " + str(complate_time))

# 删除参数文件以免硬盘爆炸
import os
def del_files_from_dir(dir,removefile=True):
    if not os.path.exists(dir):
        print("--no file can delete or path not exits ! ! !--")
    else:
        for root, directories, files in os.walk(dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                print("filepath: ", filepath)
                if(removefile == True):
                    os.remove(filepath)
                    print("removed!!!")

result_path = args.output_dir + "/" + re_model_name
del_files_from_dir(result_path,True)


if(epoch==int(args.num_train_epochs-1)):
    path = args.output_dir + '/log.txt'
    with open(path, 'a+', encoding='utf-8') as f:
        f.write("start_time: " + str(start_time) + "\n")
        f.write("complate_time: " + str(complate_time) + "\n")
        # 计算两个日期的间隔
        f.write("time spend: " + str(datetime.datetime.strptime(complate_time, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')) + "\n")
        f.write(str(args) + "\n")
        # for a in list(args):
        #     f.write(str(a) + "\n")
        for e in running_f1_list:
            f.write(str(e) + "\n")
        f.write("last_epoch=" + str(last_epoch) + " mi_f1=" + str(last_epoch_mi_f1) + " prec=" + str(last_epoch_prec) + " recall=" + str(last_epoch_recall) + " ma_f1=" + str(last_epoch_ma_f1) + "\n")
        f.write("mx_epoch=" + str(mx_epoch) + " mi_f1=" + str(mi_f1) + " prec=" + str(prec) + " recall=" + str(recall) + " ma_f1=" + str(ma_f1) + "\n\n")
    f.close()