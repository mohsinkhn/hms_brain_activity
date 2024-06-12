import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def collate_fn(batch):
    max_traces_len = max(item['traces'].shape[0] for item in batch)
    max_bboxes_len = max(item['bboxes'].shape[0] for item in batch)
    
    for item in batch:
        traces_padding = max_traces_len - item['traces'].shape[0]
        if traces_padding > 0:
            item['traces'] = torch.nn.functional.pad(item['traces'], (0, 0, 0, traces_padding))
        
        bboxes_padding = max_bboxes_len - item['bboxes'].shape[0]
        if bboxes_padding > 0:
            item['bboxes'] = torch.nn.functional.pad(item['bboxes'], (0, 0, 0, bboxes_padding))
    
    traces = torch.stack([item['traces'] for item in batch])
    bboxes = torch.stack([item['bboxes'] for item in batch])
    
    return {
        'traces': traces,
        'bboxes': bboxes
    }

def plot_traces(traces, bboxes):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.imshow(traces[:, :].T, aspect="auto", cmap="gray")
    
    for trace1, depth1, trace2, depth2 in bboxes:
        box = patches.Rectangle(
            xy=(trace1, depth1),
            width=(trace2 - trace1),
            height=(depth2 - depth1),
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(box)
    
    plt.xlabel("Trace")
    plt.ylabel("Samples")
    plt.show()

