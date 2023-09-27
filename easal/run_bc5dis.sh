MIN_LEN=4
DATA=bc5dis
A=2250

python active_learn.py \
  --active_policy mnlp \
  --bert_model ./biobert-base-cased-v1.1-pytorch/ \
  --data_dir ./biodata/BC5CDR-disease \
  --prefix $DATA-min-$MIN_LEN \
  --seed 2022 \
  --min_len $MIN_LEN \
  --A $A \
  --output_dir_ana result_ana \
  --output_dir result \
  --query_file $DATA-min-$MIN_LEN-A-$A.txt \
  --train_data train_dev_new.txt \
  --dev_data test_new.txt \
  --test_data test_new.txt \
  --label_list disease