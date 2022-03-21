import argparse
import json
from spacy.lang.en import English

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

parser = argparse.ArgumentParser()
parser.add_argument('--input_generation_file', type=str, required=True,help='input_source_file')
parser.add_argument('--input_id_file', type=str, required=True,help='input_id_file')
parser.add_argument('--input_qa_data_file', type=str, required=True,help='input_qa_data_file')
parser.add_argument('--output_file', type=str,required=True, help='output_file')
parser.add_argument('--percentage', type=float,default=1.0, help='filter percentage')



def convert_plaintext2qadataset(input_generation_file,input_qa_data_file,input_id_file,output_file,percentage):
    generations=[]
    ids=[]

    with open(input_id_file) as rf:
        for line in rf:
            ids.append(line.strip())
        rf.close()

    with open(input_generation_file) as rf:
        for line in rf:
            content = line.strip().split("\t")
            assert len(content)==2
            generation=content[0]
            lm_score=float(content[1])
            generations.append((generation,lm_score))
        rf.close()

    assert len(generations)==len(ids)

    id2generationwithscore={ids[i]:generations[i] for i in range(len(generations))}
    if percentage==1.0:
        threshold=-10000000
    else:
        sorted_d=sorted(id2generationwithscore.items(), key=lambda x: x[1][1], reverse=True)
        threshold=id2generationwithscore[sorted_d[int(percentage*len(sorted_d))][0]][1]
    datajson = json.load(open(input_qa_data_file, 'r'))
    count=0
    new_passages = []
    for passages in datajson['data']:
        new_paras = []
        for para in passages['paragraphs']:
            context = para['context']
            new_qas = []
            for qa in para['qas']:
                if id2generationwithscore[qa['id']][1] >= threshold:
                    qa['question']=id2generationwithscore[qa['id']][0]
                    new_qas.append(qa)
                    count+=1
            if len(new_qas) > 0:
                new_paras.append({'context': context, 'qas': new_qas})
        if len(new_paras) > 0:
            new_passages.append({'title': passages['title'], 'paragraphs': new_paras})
    print(count)
    # double-check
    qa_num = 0
    context_num = 0
    for passages in datajson['data']:
        for para in passages['paragraphs']:
            context_num += 1
            for qa in para['qas']:
                if len(qa['answers']) > 0:
                    qa_num += len(qa['answers'])

    print("Before filtering: #Context: %d , #QA: %d" % (context_num, qa_num))

    datajson['data'] = new_passages

    qa_num = 0
    context_num = 0
    for passages in datajson['data']:
        for para in passages['paragraphs']:
            context_num += 1
            for qa in para['qas']:
                if len(qa['answers']) > 0:
                    qa_num += len(qa['answers'])

    print("After filtering: #Context: %d , #QA: %d" % (context_num, qa_num))

    json.dump(datajson,open(output_file,'w'))


if __name__=="__main__":
    args = parser.parse_args()
    convert_plaintext2qadataset(args.input_generation_file, args.input_qa_data_file, args.input_id_file, args.output_file,args.percentage)