#!/bin/bash
if [ $# != 4 -a $# != 5 ]; then
  echo "sh gen_partitioned_data.sh graph.json index_meta output_dir shard_num [part_no]"
  exit 1;
fi
graph_json=$1
index_json=$2
dir=$3
shard_num=$4
part_no=0

if [ $# == 5 ]; then
  part_no=$5
fi

if [ $part_no == 0 ]; then
  rm -fr ${dir}
  python json2meta.py $graph_json $dir/euler.meta $shard_num
fi
python json2partdat.py $graph_json $dir/euler.meta $dir $shard_num $part_no
python json2partindex.py $index_json $graph_json $dir $shard_num $part_no

exit 0
