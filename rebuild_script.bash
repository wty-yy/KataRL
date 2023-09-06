#!/bin/bash

# 设置最大并行进程数
max_processes=20

# 初始化计数器
count=0

for i in {1..10}
do
    # ppo
    python katarl/run/ppo/ppo.py --train --wandb-track --seed $i --capture-video &
    python katarl/run/ppo/ppo.py --train --wandb-track --seed $i --capture-video --env-name Acrobot-v1 &

    # ddqn
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i &
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i --tau 0.9 &
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i --model-name mlp_jax_deeper &
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i --num_envs 1 &
    python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i --env-name Acrobot-v1 &

    # a2c
    python katarl/run/a2c/a2c.py --train --wandb-track --seed $i &
    python katarl/run/a2c/a2c.py --train --wandb-track --seed $i --env-name Acrobot-v1 &

    # 增加计数器
    ((count++))

    # 如果达到最大并行进程数，等待它们完成
    if [ $count -eq $max_processes ]; then
        wait
        count=0  # 重置计数器
    fi
done

# 等待所有后台任务完成
wait

