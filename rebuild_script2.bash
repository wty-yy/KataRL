#!/bin/bash

# 设置最大并行进程数
max_processes=20

# 创建一个函数，用于运行每个任务
run_task() {
    i="$1"
    # ddqn
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i --num-envs 1
}

export -f run_task

# 使用 parallel 命令并发运行任务
seq 1 10 | parallel -j $max_processes run_task {}

# 等待所有任务完成
wait

