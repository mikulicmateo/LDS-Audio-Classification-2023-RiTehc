from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os


class IRMASDataset(Dataset):

    def __init__(self, annotations_file, project_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.project_dir = project_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        return os.path.join(self.project_dir, self.annotations.iloc[index, 1])

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 3]


# Check if data is loaded properly.
if __name__ == "__main__":
    ANNOTATIONS_FILE = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/IRMAS_Training_Data/training_annotation_file.csv'
    PROJECT_DIR = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc'
    ds = IRMASDataset(ANNOTATIONS_FILE, PROJECT_DIR)

    print(f'There are {len(ds)} samples')
    # signal, label = ds[0]
