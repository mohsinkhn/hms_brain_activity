import torch
from torch.utils.data import Dataset

import numpy as np
import os
import segyio

class GPRDataset(Dataset):
    def __init__(self, root_dir, window_size, stride, bboxes_fixed_size):
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.bboxes_fixed_size = bboxes_fixed_size
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.sgy')]
        self.subfiles = self._generate_subfiles()

    def _generate_subfiles(self):
        subfiles = []
        for segy_file in self.files:
            base_name = os.path.splitext(segy_file)[0]
            annotation_file = base_name + '.txt'
            segy_path = os.path.join(self.root_dir, segy_file)
            traces = []
            with segyio.open(segy_path, "r", endian='little', strict=False) as segy_data:
                for i in range(segy_data.tracecount):
                    traces.append(segy_data.trace[i])
                traces = np.stack(traces)
            traces = torch.tensor(traces, dtype=torch.float32)

            bboxes = []
            with open(os.path.join(self.root_dir, annotation_file), 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    bbox = list(map(int, parts[:4]))
                    bboxes.append(bbox)
            bboxes = torch.tensor(bboxes, dtype=torch.float64)

            num_traces = traces.shape[0]
            start = 0
            while start < num_traces:
                end = start + self.window_size
                if end > num_traces:
                    end = num_traces
                sub_traces = traces[start:end]

                if sub_traces.shape[0] < self.window_size:
                    padding = self.window_size - sub_traces.shape[0]
                    sub_traces = torch.nn.functional.pad(sub_traces, (0, 0, 0, padding))

                sub_bboxes = bboxes[(bboxes[:, 0] >= start) & (bboxes[:, 2] <= end)].clone()
                sub_bboxes[:, [0, 2]] -= start  

                # Pad or truncate bboxes to the fixed size
                if sub_bboxes.shape[0] > self.bboxes_fixed_size:
                    sub_bboxes = sub_bboxes[:self.bboxes_fixed_size]
                elif sub_bboxes.shape[0] < self.bboxes_fixed_size:
                    padding = self.bboxes_fixed_size - sub_bboxes.shape[0]
                    sub_bboxes = torch.nn.functional.pad(sub_bboxes, (0, 0, 0, padding))

                subfiles.append((sub_traces, sub_bboxes))

                if end == num_traces:
                    break

                start += self.stride

        return subfiles

    def __len__(self):
        return len(self.subfiles)

    def __getitem__(self, idx):
        traces, bboxes = self.subfiles[idx]
        sample = {
            'traces': traces,
            'bboxes': bboxes
        }
        return sample
