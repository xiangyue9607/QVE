#! /bin/bash

cd data

declare -a arr=("SQuAD" "NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web")

##downloading the datasets from the MRQA 2019.
##We use the dev set as the test set

for dataset_name in "${arr[@]}"; do
  echo "Downloading dataset: $dataset_name ..."
  wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/"$dataset_name".jsonl.gz -O "$dataset_name".train.jsonl.gz
  wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/"$dataset_name".jsonl.gz -O "$dataset_name".test.jsonl.gz

  gzip -d "$dataset_name".train.jsonl.gz

  ##format dataset to SQuAD-like json file
  python convert_jsonl2json.py \
  -input "$dataset_name".train.jsonl \
  -output "$dataset_name".train.json

  gzip -d "$dataset_name".test.jsonl.gz
  python convert_jsonl2json.py \
  -input "$dataset_name".test.jsonl \
  -output "$dataset_name".test.json

  rm "$dataset_name".train.jsonl
  rm "$dataset_name".test.jsonl

  ##For all the target domain datasets {"NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web"}
  ##We sample 1000 QAs from the training as the dev set
  if [ "$dataset_name" != "SQuAD" ]; then
    echo "Sampling dev set from the training set..."
    python split_data_num.py \
    --in_file "$dataset_name".train.json \
    --out_file_dev "$dataset_name".sample.dev.json \
    --out_file_train "$dataset_name".sample.train.json \
    --num 1000
  fi
done
