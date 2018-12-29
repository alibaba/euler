#!/usr/bin/env bash

NUM_PSES=2
NUM_WORKERS=2

PS_HOSTS=""
for I in $(seq $(($NUM_PSES - 1)) -1 0)
do
  PS_HOST="localhost:$((1999 - $I))"
  if [ "$PS_HOSTS" == "" ]; then
    PS_HOSTS="$PS_HOST"
  else
    PS_HOSTS="$PS_HOSTS,$PS_HOST"
  fi
done

WORKER_HOSTS=""
for I in $(seq 0 $(($NUM_WORKERS - 1)))
do
  WORKER_HOST="localhost:$((2000 + $I))"
  if [ "$WORKER_HOSTS" == "" ]; then
    WORKER_HOSTS="$WORKER_HOST"
  else
    WORKER_HOSTS="$WORKER_HOSTS,$WORKER_HOST"
  fi
done

for I in $(seq 0 $(($NUM_PSES - 1)))
do
  python -m tf_euler \
    --ps_hosts=$PS_HOSTS \
    --worker_hosts=$WORKER_HOSTS \
    --job_name=ps --task_index=$I &> /tmp/log.ps.$I &
done

for I in $(seq 0 $(($NUM_WORKERS - 1)))
do
  python -m tf_euler \
    --ps_hosts=$PS_HOSTS \
    --worker_hosts=$WORKER_HOSTS \
    --job_name=worker --task_index=$I $@ &> /tmp/log.worker.$I &
  WORKERS[$I]=$!
done

trap 'kill $(jobs -p) 2> /dev/null; exit' SIGINT SIGTERM

wait ${WORKERS[*]}

kill $(jobs -p) 2> /dev/null
