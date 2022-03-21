import json_lines
import json
from argparse import ArgumentParser

data = []
note = 0
count = 0

parser = ArgumentParser()
parser.add_argument("-input", dest="input", default="TriviaQA-web.jsonl")
parser.add_argument("-output", dest="output", default="TriviaQA-web.json")
args = parser.parse_args()

with open(args.input, 'rb') as f, open(args.output, "w") as out:
    for item in json_lines.reader(f):
        data.append(item)

    new_data = {'data': [], 'version': 'v0.1'}

    starter = 0
    if 'header' in data[0]:
        starter = 1
    else:
        data = data[0]

    for i in range(starter, len(data)):
        new_data['data'].append({'title': str(i), 'paragraphs': []})

        context = data[i]['context']
        len1=len(context)
        context = context.replace(u'\u00A0', ' ')
        assert len(context)==len1
        new_data['data'][note]['paragraphs'].append({'context': context, 'qas': []})

        for k in range(len(data[i]['qas'])):
            answers = []
            for p in range(len(data[i]['qas'][k]['detected_answers'])):
                text = data[i]['qas'][k]['detected_answers'][p]['text']
                answer_index = data[i]['qas'][k]['detected_answers'][p]['char_spans'][0]
                answers.append({'answer_start': answer_index[0], 'text': context[answer_index[0]:answer_index[1]+1]})
            if len(answers) > 0:
                new_data['data'][note]['paragraphs'][0]['qas'].append(
                    {'question': data[i]['qas'][k]['question'], 'id': data[i]['qas'][k]['qid'], 'answers': answers})
                count += 1
        note += 1

    print(count)
    new_data["len"] = count
    json.dump(new_data, out)
