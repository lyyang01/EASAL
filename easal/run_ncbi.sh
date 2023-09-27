MIN_LEN=4
DATA=ncbi
A=1500

python active_learn.py \
  --active_policy mnlp \
  --bert_model ./biobert-base-cased-v1.1-pytorch/ \
  --data_dir ./biodata \
  --prefix $DATA-min-$MIN_LEN \
  --seed 2022 \
  --min_len $MIN_LEN \
  --A $A \
  --output_dir_ana result_ana \
  --output_dir result \
  --query_file $DATA-min-$MIN_LEN-A-$A.txt \
  --train_data train.txt \
  --dev_data test.txt \
  --test_data test.txt \
  --label_list disease