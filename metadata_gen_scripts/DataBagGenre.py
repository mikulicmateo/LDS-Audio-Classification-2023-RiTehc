import pandas as pd
import numpy as np


class DataBagGenre:
    def __init__(self, default_train_annotation_file):
        self.annotation_file = pd.read_csv(default_train_annotation_file)
        self.genres = ['cla', 'jaz_blu', 'pop_roc', 'cou_fol', 'lat_sou']
        self.bag = self.create_bag_dict()

    def get_files_for_genre(self, label, genre, ann_file):
        genre_files = ann_file.loc[(ann_file[label] == 1) & (ann_file['genre'] == genre)]['path'].to_numpy()
        return genre_files

    def get_genre_dict(self, label, ann_file):
        genres = {}
        for genre in self.genres:
            # get all filepaths for this label and genre
            genres.update({genre: self.get_files_for_genre(label, genre, ann_file)})

        return genres

    def get_files_for_label(self, label, ann_file):
        file_paths_dict_for_label = self.get_genre_dict(label, ann_file)

        for key in file_paths_dict_for_label.keys():
            np.random.shuffle(file_paths_dict_for_label[key])

        return file_paths_dict_for_label

    def create_bag_dict(self):
        bag = {}
        for column in self.annotation_file.columns:
            if column != 'path' and column != 'Unnamed: 0':
                bag.update({column: self.get_files_for_label(column, self.annotation_file)})

        #print(bag['tru']['cla'])

        return bag

    def get_bag_item(self, label, genre):
        # if a genre does not have that particular instrument return none
        if self.bag[label][genre].size == 0:
            return None

        item = self.bag[label][genre][0]
        np.delete(self.bag[label][genre], 0)
        if self.bag[label][genre].size == 0:
            self.repopulate_bag_for_genre(label, genre)
        np.random.shuffle(self.bag[label][genre])

        return item

    def repopulate_bag_for_label(self, label):
        self.bag[label] = self.get_files_for_label(label, self.annotation_file)

    def repopulate_bag_for_genre(self, label, genre):
        self.bag[label][genre] = self.get_files_for_genre(label, genre, self.annotation_file)