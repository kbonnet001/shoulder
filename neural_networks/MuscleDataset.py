import random
from torch.utils.data import Dataset

class MuscleDataset(Dataset):
    """Create a personalize Dataset"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def remove_random_items(self, num_items_to_remove):
        if num_items_to_remove >= len(self.y):
            raise ValueError("Number of items to remove exceeds or equals the dataset size")

        indices_to_remove = random.sample(range(len(self.y)), num_items_to_remove)
        indices_to_keep = list(set(range(len(self.y))) - set(indices_to_remove))

        self.X = [self.X[i] for i in indices_to_keep]
        self.y = [self.y[i] for i in indices_to_keep]