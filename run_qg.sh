#! /bin/bash

dataset_name=TriviaQA-web
n_GPUS=1
SOURCE_QG_CKPT_DIR=checkpoints/QG_source_only
TARGET_DEV_QG_CKPT_DIR=checkpoints/QG_source_then_"$dataset_name"_dev

mkdir checkpoints

if ! [ -d "$SOURCE_QG_CKPT_DIR" ]; then
  echo "##########################################################################"
  echo "$SOURCE_QG_CKPT_DIR does not exist. Begin QG Training."
  echo "##########################################################################"
  echo "Preprocessing source files (SQuAD) for QG..."
  echo "##########################################################################"
  mkdir "data/SQuAD_QG"
  python data/convert_qadatasets_to_plaintext.py \
    --input_file data/SQuAD.train.json \
    --output_srcfile data/SQuAD_QG/train.source \
    --output_tgtfile data/SQuAD_QG/train.target \
    --output_idfile data/SQuAD_QG/train.id \
    --tokenizer_path facebook/bart-base

  python data/convert_qadatasets_to_plaintext.py \
    --input_file data/SQuAD.test.json \
    --output_srcfile data/SQuAD_QG/val.source \
    --output_tgtfile data/SQuAD_QG/val.target \
    --output_idfile data/SQuAD_QG/val.id \
    --tokenizer_path facebook/bart-base

  echo "##########################################################################"
  echo "Pretrain QG model on SQuAD"
  echo "##########################################################################"
  python QG/finetune.py \
    --learning_rate 3e-5 \
    --do_train \
    --data_dir data/SQuAD_QG \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --output_dir $SOURCE_QG_CKPT_DIR \
    --gpus $n_GPUS \
    --num_train_epochs 3 \
    --num_processes 16 \
    --model_name_or_path facebook/bart-base \
    --task translation \
    --overwrite_output_dir
fi


if [ -d "$TARGET_DEV_QG_CKPT_DIR" ]; then
  echo "##########################################################################"
  echo "$TARGET_DEV_QG_CKPT_DIR exists. Skip Training."
  echo "##########################################################################"
else
  echo "##########################################################################"
  echo "Finetuning source QG on the target ($dataset_name) dev set"
  echo "##########################################################################"

  mkdir data/"$dataset_name"_QG/
  
  python data/convert_qadatasets_to_plaintext.py \
  --input_file data/"$dataset_name".sample.dev.json \
  --output_srcfile data/"$dataset_name"_QG/train.source \
  --output_tgtfile data/"$dataset_name"_QG/train.target \
  --output_idfile data/"$dataset_name"_QG/train.id \
  --tokenizer_path facebook/bart-base

  python data/convert_qadatasets_to_plaintext.py \
  --input_file data/"$dataset_name".test.json \
  --output_srcfile data/"$dataset_name"_QG/val.source \
  --output_tgtfile data/"$dataset_name"_QG/val.target \
  --output_idfile data/"$dataset_name"_QG/val.id \
  --tokenizer_path facebook/bart-base

  python QG/finetune.py \
  --learning_rate 3e-5 \
  --do_train \
  --data_dir data/"$dataset_name"_QG \
  --train_batch_size 4 \
  --eval_batch_size 8 \
  --output_dir $TARGET_DEV_QG_CKPT_DIR \
  --gpus $n_GPUS \
  --num_train_epochs 3 \
  --num_processes 16 \
  --model_name_or_path $SOURCE_QG_CKPT_DIR/best_tfmr \
  --task translation \
  --overwrite_output_dir
fi



echo "##########################################################################"
echo "Preprocessing files for QG on $dataset_name"
echo "##########################################################################"
mkdir data/"$dataset_name"_QG

python data/convert_qadatasets_to_plaintext.py \
--input_file data/"$dataset_name".train.json \
--output_srcfile data/"$dataset_name"_QG/test.source \
--output_tgtfile data/"$dataset_name"_QG/test.target \
--output_idfile data/"$dataset_name"_QG/test.id \
--tokenizer_path facebook/bart-base


echo "##########################################################################"
echo "Generate synthetic QAs on $dataset_name training contexts based on $TARGET_DEV_QG_CKPT_DIR"
echo "##########################################################################"

python -m torch.distributed.launch --nproc_per_node=$n_GPUS \
  QG/run_distributed_eval.py \
  --model_name $TARGET_DEV_QG_CKPT_DIR/best_tfmr \
  --data_dir data/"$dataset_name"_QG/ \
  --save_dir data/"$dataset_name"_QG/ \
  --task translation \
  --bs 20


echo "##########################################################################"
echo "Converting Generations to QA file"
echo "##########################################################################"
mv data/"$dataset_name"_QG/test_generations.txt data/"$dataset_name"_QG/test_generations_by_finetunedqg.txt

python data/convert_plaintext_to_qadataset.py \
  --input_generation_file data/"$dataset_name"_QG/test_generations_by_finetunedqg.txt \
  --input_id_file data/"$dataset_name"_QG/test.id \
  --input_qa_data_file data/"$dataset_name".train.json \
  --output_file data/"$dataset_name"_QG/"$dataset_name".train.targetfinedtuned.gen.json







