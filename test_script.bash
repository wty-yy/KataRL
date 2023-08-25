#!/bin/bash
# conda activate jax
end=$1

for i in $(seq 1 $end)
do
    echo "run time: $i/$end"
    # python run/DQN/dqn.py --train --wandb-track
    # python run/A2C/a2c.py --train --wandb-track
    python run/PPO/ppo.py --train --wandb-track --capture-video
done
