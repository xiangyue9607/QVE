import argparse
import json
from tqdm import tqdm
import random
import re
from transformers import BartTokenizer



parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True,help='input_file')
parser.add_argument('--output_srcfile', type=str,required=True, help='output_srcfile')
parser.add_argument('--output_tgtfile', type=str,required=True, help='output_tgtfile')
parser.add_argument('--output_idfile', type=str,required=True, help='output_idfile')
parser.add_argument('--max_source_length', type=int, default=512)
parser.add_argument('--min_source_length', type=int, default=1)
parser.add_argument('--doc_stride', type=int, default=128)
parser.add_argument('--tokenizer_path', type=str, required=True)


def chunk_stride(lst, n, stride):
    yield_flag = True
    for i in range(0, len(lst), stride):
        if yield_flag:
            if i + n >= len(lst):
                yield_flag = False
            yield lst[i: i + n]


def convert_squad2plaintext(input_file,output_srcfile, output_tgtfile, output_idfile,tokenizer):
    srcs=[]
    tgts=[]
    ids=[]
    data_json= json.load(open(input_file, 'r'))
    for paras in tqdm(data_json['data']):
        for para in paras['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                question=qa['question']
                answer=qa['answers'][0]
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer['text'])
                tag_context = context[:answer_start] + " <hl> " + \
                              context[answer_start:answer_end] \
                              + " <hl> " + context[answer_end:]

                tag_context = " ".join(tag_context.split())
                question = " ".join(question.split())

                src = tokenizer.encode(tag_context, add_special_tokens=False, max_length=100000, truncation=True)
                max_len = args.max_source_length - 2

                if len(src) < args.min_source_length-2:
                    continue

                if len(src) > max_len:
                    # ensure that src includes answer
                    ans_end_index = src.index(tokenizer.additional_special_tokens_ids[0])
                    assert ans_end_index >= 0
                    for jj, con_chunk in enumerate(chunk_stride(src, max_len, args.doc_stride)):
                        if jj * args.doc_stride + max_len > ans_end_index:
                            src = con_chunk
                            break

                tag_context=tokenizer.decode(src)

                tag_context = "generate question: " + tag_context
                srcs.append(tag_context)
                tgts.append(question)
                ids.append(qa['id'])


    assert len(srcs) == len(tgts)
    print(len(srcs))

    with open(output_srcfile, 'w') as fout:
        for src in srcs:
            fout.write(src + "\n")
        fout.close()

    with open(output_tgtfile, 'w') as fout:
        for tgt in tgts:
            fout.write(tgt + "\n")
        fout.close()

    with open(output_idfile, 'w') as fout:
        for qid in ids:
            fout.write(qid + "\n")
        fout.close()


if __name__=="__main__":
    args = parser.parse_args()
    random.seed(1)

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)
    special_tokens_dict = {'additional_special_tokens': ['<hl>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    convert_squad2plaintext(args.input_file,args.output_srcfile,args.output_tgtfile,args.output_idfile,tokenizer)

