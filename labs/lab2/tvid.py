from torch.utils import data


class TvidDataset(data.Dataset):
    # TODO Implement the dataset class inherited
    # from `torch.utils.data.Dataset`.
    # tips: Use `transforms`.

    ...

    # End of todo


if __name__ == '__main__':
    dataset = TvidDataset(root='~/data/tiny_vid', mode='train')
    import pdb;

    pdb.set_trace()
