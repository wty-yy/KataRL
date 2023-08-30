#!/bin/bash
# conda activate jax
end=$1

for i in $(seq 1 $end)
do
    echo "run time: $i/$end"
    # python run/dqn/dqn.py --train --wandb-track
    # python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i
    python katarl/run/a2c/a2c.py --train --wandb-track --seed $i
    # python run/PPO/ppo.py --train --wandb-track --capture-video
done

