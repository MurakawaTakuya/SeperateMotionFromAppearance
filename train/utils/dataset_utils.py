"""
Dataset-related utility functions.
"""
from train.utils.dataset import VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def get_train_dataset(dataset_types, train_data, tokenizer):
    """Get training dataset based on types."""
    train_datasets = []

    # Loop through all available datasets, get the name, then add to list of data to process.
    for DataSet in [VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset]:
        for dataset in dataset_types:
            if dataset == DataSet.__getname__():
                train_datasets.append(
                    DataSet(**train_data, tokenizer=tokenizer))

    if len(train_datasets) > 0:
        return train_datasets
    else:
        raise ValueError(
            "Dataset type not found: 'json', 'single_video', 'folder', 'image'")


def extend_datasets(datasets, dataset_items, extend=False):
    """Extend datasets that are less than the greatest one."""
    biggest_data_len = max(x.__len__() for x in datasets)
    extended = []
    for dataset in datasets:
        if dataset.__len__() == 0:
            del dataset
            continue
        if dataset.__len__() < biggest_data_len:
            for item in dataset_items:
                if extend and item not in extended and hasattr(dataset, item):
                    print(f"Extending {item}")

                    value = getattr(dataset, item)
                    value *= biggest_data_len
                    value = value[:biggest_data_len]

                    setattr(dataset, item, value)

                    print(f"New {item} dataset length: {dataset.__len__()}")
                    extended.append(item)
