import plotly.graph_objects as go
import pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class MouseMoveDataset(Dataset):
    def __init__(self, input_sequences, targets):
        self.input_sequences = input_sequences
        self.targets = targets

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.targets[idx]

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.stack(yy), torch.tensor(x_lens)

def getDataloader(norm_input_trajectories, norm_buttonTargets, batch_size):
    dataset = MouseMoveDataset(norm_input_trajectories, norm_buttonTargets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    return dataloader

def visuallyVertifyDataloader(dataloader, dataset, showNumBatches=1):
    fig = go.Figure()
    for i, data in enumerate(dataloader, 0): 
        _input_trajectories_padded, _buttonTargets, trajectoryLengths = data
        if i == showNumBatches:
            break
        for ii in range(len(_input_trajectories_padded)):
            traj = _input_trajectories_padded[ii] * dataset.std_traj + dataset.mean_traj
            traj = traj[:trajectoryLengths[ii]]
            df_sequence = pd.DataFrame(traj, columns=['dx','dy'])
            df_sequence['velocity'] = np.sqrt(df_sequence['dx']**2 + df_sequence['dy']**2) / dataset.FIXED_TIMESTEP
            df_target = pd.DataFrame(_buttonTargets[ii] * dataset.std_button + dataset.mean_button, columns=[dataset.targetColumns])
            sequence_id = 0
            dataset.SHOW_ONE = True
            df_abs = dataset.convertToAbsolute(df_sequence, df_target)

            fig.add_trace(go.Scatter(x=df_abs['x'], y=df_abs['y'],
                    mode='lines+markers',
                    showlegend=False,
                    marker=dict(
                        size=5, 
                        # symbol= "arrow-bar-up", angleref="previous",
                        # size=15,
                        # color='grey',),
                        color=df_abs['velocity'], colorscale='Viridis', showscale=False, 
                        colorbar=None,
                        # colorbar=dict(title="Velocity")),
                    )))
    fig.update_layout(
        title=str(showNumBatches) + " batches of dataloader, <br> data has been unnormalized to verify correctness",
        width=500,
        height=500,)
    fig.show()