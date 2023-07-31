# -*- coding: utf-8 -*-
'''
generate gif file from frames.
用于生成gymnasium环境的gif动态图
'''

from matplotlib import animation
import matplotlib.pyplot as plt
from pathlib import Path

def save_frames_as_gif(frames, fname, path:Path, **kargs):

    #Mess with this to change frame size
    fig, ax = plt.subplots(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    if 'step' in kargs.keys():
        ax.set_title(f"step:{kargs['step']}")

    patch = ax.imshow(frames[0])
    ax.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path.joinpath(fname + ".gif"), writer='pillow', fps=60)


if __name__ == '__main__':
    pass