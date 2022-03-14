# Question Value Estimation (QVE)

## Introduction
This repository provides code for the ACL2022 paper "[Synthetic Question Value Estimation for Domain Adaptation of Question Answering](.)"

```bib
@inproceedings{yue2022qve,
 title={Synthetic Question Value Estimation for Domain Adaptation of Question Answering},
 author={Xiang Yue and Ziyu Yao and Huan Sun},
 booktitle={ACL},
 year={2022}
}
```

## Set up
Run the following commands to clone the repository and install requirements. 
It requires Python 3.7 or higher. 
It also requires installing [Huggingface transformers](https://github.com/huggingface/transformers) version 3.3.1.
The other dependencies are listed in requirements.txt. 
```shell script
$ git clone https://github.com/xiangyue9607/QVE.git
$ pip install -r requirements.txt 
```

## Preparing Data
We use "SQuAD" as the source dataset and "NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web" as target datasets. All the datasets can be downloaded from [MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019). We use the original dev set as the test set and sample a limited number (by default: 1000) of QA pairs from the training as the dev set.

```shell script
$ sh download.sh
```

## Base QG Model Training
We consider a semi-supervised domain adaptation setting where we first pretrain the QG model on the source domain and then finetune it on the limited number of target dev set.

```shell script
$ sh run_qg.sh
```

## Base QA Model Training
For QA model, we also first pretrain it on the source and then finetune it on the target dev.
```shell script
$ sh run_qa_baseline.sh
```

## QVE Training
```shell script
$ sh run_qa_qve.sh
```
