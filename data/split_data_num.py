import json
import random
from argparse import ArgumentParser
from numpy.random import default_rng

parser = ArgumentParser()
parser.add_argument("--in_file", type=str, default="data/NewsQA.train.json",)
parser.add_argument("--out_file_dev", type=str,  default="dataNewsQA.sample.dev.json")
parser.add_argument("--out_file_train", type=str, default="data/NewsQA.sample.train.json")
parser.add_argument("--num", type=int, default=1000, required=False)
parser.add_argument("--seed", type=int, default=42, required=False)

args = parser.parse_args()


def subsample_dataset_random(data_json, sample_num=1000, seed=55):

    total = 0
    context_num=0
    id_lists=[]
    for paras in data_json['data']:
        for para in paras['paragraphs']:
            context_num+=1
            qa_num = len(para['qas'])
            id_lists+=[qa['id'] for qa in para['qas']]
            total += qa_num
    print('Total QA Num: %d, Total Context: %d' % (total,context_num))

    random.seed(seed)
    rng = default_rng()
    sampled_list = list(rng.choice(id_lists, size=sample_num,replace=False))
    new_passages_dev = []
    new_passages_train=[]

    for passages in data_json['data']:
        new_paras_dev = []
        new_paras_train = []

        for para in passages['paragraphs']:
            context = para['context']
            new_qas_dev = []
            new_qas_train = []

            for qa in para['qas']:
                if qa['id'] in sampled_list:
                    new_qas_dev.append(qa)
                else:
                    new_qas_train.append(qa)

            if len(new_qas_dev) > 0:
                new_paras_dev.append({'context': context, 'qas': new_qas_dev})
            if len(new_qas_train) > 0:
                new_paras_train.append({'context': context, 'qas': new_qas_train})

        if len(new_paras_dev) > 0:
            new_passages_dev.append({'title': passages['title'], 'paragraphs': new_paras_dev})

        if len(new_paras_train) > 0:
            new_passages_train.append({'title': passages['title'], 'paragraphs': new_paras_train})

    dev_data_json = {'data': new_passages_dev, 'version': data_json['version']}
    train_data_json = {'data': new_passages_train, 'version': data_json['version']}

    total = 0
    context_num=0
    for paras in dev_data_json['data']:
        for para in paras['paragraphs']:
            context_num+=1
            qa_num = len(para['qas'])
            id_lists+=[qa['id'] for qa in para['qas']]
            total += qa_num
    print('Sample Dev QA Num: %d, Total Context: %d' % (total,context_num))

    total = 0
    context_num = 0
    for paras in train_data_json['data']:
        for para in paras['paragraphs']:
            context_num += 1
            qa_num = len(para['qas'])
            id_lists += [qa['id'] for qa in para['qas']]
            total += qa_num
    print('Sample Train QA Num: %d, Total Context: %d' % (total, context_num))

    return train_data_json,dev_data_json

def main(args):

    dataset = json.load(open(args.in_file, 'r'))

    train_data_json,dev_data_json=subsample_dataset_random(dataset, args.num, args.seed)

    json.dump(train_data_json, open(args.out_file_train, 'w'))
    json.dump(dev_data_json, open(args.out_file_dev, 'w'))


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
