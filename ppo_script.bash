#!/bin/bash

python katarl/run/ppo/ppo_atari.py --train --wandb-track --capture-video --flag-anneal-reward yes

python katarl/run/ppo/ppo_atari.py --train --wandb-track --capture-video --flag-anneal-reward no

