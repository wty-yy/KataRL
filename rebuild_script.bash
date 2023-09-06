#!/bin/bash

# 设置最大并行进程数
max_processes=20

# 创建一个函数，用于运行每个任务
run_task() {
    i="$1"
    # ppo
    python katarl/run/ppo/ppo.py --train --wandb-track --seed $i --capture-video
    python katarl/run/ppo/ppo.py --train --wandb-track --seed $i --capture-video --env-name Acrobot-v1

    # ddqn
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i --tau 0.9
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i --model-name mlp_jax_deeper
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i --num-envs 1
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i --env-name Acrobot-v1

    # a2c
    python katarl/run/a2c/a2c.py --train --wandb-track --seed $i
    python katarl/run/a2c/a2c.py --train --wandb-track --seed $i --env-name Acrobot-v1
}

export -f run_task

# 使用 parallel 命令并发运行任务
seq 1 10 | parallel -j $max_processes run_task {}

# 等待所有任务完成
wait

