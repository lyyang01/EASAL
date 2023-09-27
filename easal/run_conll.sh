MIN_LEN=4
DATA=conll
A=2000
SEED=2022
#ori 2022

python active_learn.py \
  --active_policy mnlp \
  --bert_model ./bert-base-cased/ \
  --data_dir ./data \
  --prefix $DATA-min-$MIN_LEN \
  --seed $SEED \
  --min_len $MIN_LEN \
  --A $A \
  --output_dir_ana result_ana \
  --output_dir result \
  --query_file $DATA-min-$MIN_LEN-A-$A.txt \
  --train_data train.txt \
  --dev_data test.txt \
  --test_data test.txt \
  --label_list PER LOC ORG MISC \
  --do_sampling