from torch.utils.data import DataLoader

import dataset


def get_dataloader(inp_path, lab_path, train_scale, kernel_size, batch_size):

    train_dataset = dataset.PDBDataset(inp_path, lab_path, kernel_size=kernel_size, train_scale=train_scale, train=True)
    valid_dataset = dataset.PDBDataset(inp_path, lab_path, kernel_size=kernel_size, train_scale=train_scale, train=False)
    
    print('Length of trianing set: {}'.format(len(train_dataset)))
    print('Length of validation set: {}'.format(len(valid_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, valid_dataloader

