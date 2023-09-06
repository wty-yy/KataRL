#!/bin/bash
# conda activate jax
end=$1

for i in $(seq 2 $end)
do
    echo "run time: $i/$end"
    # python run/dqn/dqn.py --train --wandb-track
    # python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i
    # python katarl/run/dqn/ddqn.py --train --wandb-track --seed $i --env-name Acrobot-v1
    # python katarl/run/a2c/a2c.py --train --wandb-track --seed $i --env-name Acrobot-v1
    # python katarl/run/ppo/ppo.py --train --wandb-track --seed $i --capture-video
    # python katarl/run/ppo/ppo.py --train --wandb-track --seed $i --capture-video
    # python katarl/run/ppo/ppo.py --env-name Acrobot-v1 --train --wandb-track --seed $i --capture-video
    # python katarl/run/sac/sac.py --train --wandb-track --seed $i --env-name Acrobot-v1  --flag-autotune-alpha no
    python katarl/run/ppo/ppo_atari.py --train --wandb-track --seed $i --capture-video
done

