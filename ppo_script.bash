#!/bin/bash

python katarl/run/ppo/ppo_atari.py --train --seed 2 --wandb-track --capture-video --flag-anneal-reward yes
python katarl/run/ppo/ppo_atari.py --train --seed 3 --wandb-track --capture-video --flag-anneal-reward yes

