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
