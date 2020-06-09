set -eux
./dist_tf_euler.sh \
  --data_dir /tmp/ppi_data/ \
  --euler_zk_addr 127.0.0.1:2181 --euler_zk_path /euler-2.0ppi \
  --max_id 56944 --feature_idx f1 --feature_dim 50 --label_idx label --label_dim 121 \
  --model graphsage_supervised --mode train
