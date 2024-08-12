import random
from torch.utils.data import Dataset

class MuscleDataset(Dataset):
    """ Custom dataset for muscle data.

    Args:
        X (list or array-like): Input features.
        y (list or array-like): Target labels.
    """

    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError("Features and labels must have the same length.")
        self.X = X
        self.y = y

    def __len__(self):
        """ Return the number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """ Retrieve the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (feature, label) for the sample at index `idx`.
        """
        return self.X[idx], self.y[idx]

    def remove_random_items(self, num_items_to_remove):
        """ Remove a specified number of random items from the dataset.

        Args:
            num_items_to_remove (int): Number of items to remove.

        Raises:
            ValueError: If `num_items_to_remove` is greater than or equal to the dataset size.
        """
        if num_items_to_remove >= len(self.y):
            raise ValueError("Number of items to remove exceeds or equals the dataset size")

        indices_to_remove = random.sample(range(len(self.y)), num_items_to_remove)
        indices_to_keep = list(set(range(len(self.y))) - set(indices_to_remove))

        self.X = [self.X[i] for i in indices_to_keep]
        self.y = [self.y[i] for i in indices_to_keep]
        