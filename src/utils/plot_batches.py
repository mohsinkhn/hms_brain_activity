import matplotlib.pyplot as plt
import numpy as np


def plot_batch(batch_x, batch_y, preds=None, save_path=None):
    batch_x = batch_x[:8]
    batch_y = batch_y[:8]
    if preds is not None:
        preds = preds[:8]
    fig, axs = plt.subplots(2, 4, figsize=(12, 12))
    for i in range(8):
        ax = axs[i // 4, i % 4]
        offset = 0
        for j in range(16):
            if j != 0:
                offset -= batch_x[i, :, j].min()
            ax.plot(batch_x[i, :, j] + offset)
            offset += batch_x[i, :, j].max() + 1
        y_str = " ".join([f"{y:.1f}" for y in batch_y[i]])
        ax.set_title(f"y={y_str}")
        if preds is not None:
            pred_str = " ".join([f"{y:.1f}" for y in preds[i]])
            ax.text(0, offset, f"pred={pred_str}")
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_zoomed_batch(batch_x, batch_y, start=0, n=100, save_path=None):
    batch_x = batch_x[:8]
    batch_y = batch_y[:8]
    fig, axs = plt.subplots(2, 4, figsize=(12, 12))
    for i in range(8):
        ax = axs[i // 4, i % 4]
        offset = 0
        for j in range(16):
            if j != 0:
                offset -= batch_x[i, start : start + n, j].min()
            ax.plot(batch_x[i, start : start + n, j] + offset)
            offset += batch_x[i, start : start + n, j].max() + 1
        y_str = " ".join([f"{y:.1f}" for y in batch_y[i]])
        ax.set_title(f"y={y_str}")
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
