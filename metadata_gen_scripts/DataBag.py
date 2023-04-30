import pandas as pd
import numpy as np


class DataBag:
    def __init__(self, default_train_annotation_file):
        self.annotation_file = pd.read_csv(default_train_annotation_file)
        self.bag = self.create_bag_dict()

    def get_files_for_label(self, label, ann_file):
        file_paths_for_label = ann_file.loc[ann_file[label] == 1]['path'].to_numpy()
        np.random.shuffle(file_paths_for_label)
        return file_paths_for_label

    def create_bag_dict(self):
        bag = {}
        for column in self.annotation_file.columns:
            if column != 'path' and column != 'Unnamed: 0':
                bag.update({column: self.get_files_for_label(column, self.annotation_file)})
        print('Created bag for keys: ', bag.keys())
        return bag

    def get_bag_item(self, label):
        item = self.bag[label][0]
        np.delete(self.bag[label], 0)
        if self.bag[label].size == 0:
            self.repopulate_bag_for_label(label)
        np.random.shuffle(self.bag[label])
        return item

    def repopulate_bag_for_label(self, label):
        self.bag[label] = self.get_files_for_label(label, self.annotation_file)
