import pandas as pd
import os

class Data:
    def __init__(self, annotations_path, data_path, modalities):
        self.annotations_path = annotations_path
        self.temporary_path = os.path.join(data_path, "temp")
        self.data_path = data_path
        self.multimodal_path = os.path.join(data_path, "raw", "S04_1.pkl")
        self.modalities = modalities
        self.train, self.test = self.get_train_test_split(save=True)
    
    def get_multimodal(self, save=False):
        train = self.train.loc[self.train["subject"]=="S04"]
        test = self.test.loc[self.test["subject"]=="S04"]
        if save:
            self.save_data(os.path.join(self.temporary_path, "multimodal_train.pkl"), train)
            self.save_data(os.path.join(self.temporary_path, "multimodal_test.pkl"), test)
        return train, test
        
    def get_train_test_split(self, save=False):
        train=self.merge("train")
        test=self.merge("test")
        if save:
            self.save_data(os.path.join(self.temporary_path, "train.pkl"), train)
            self.save_data(os.path.join(self.temporary_path, "test.pkl"), test)
        return train, test
        
    def merge(self, mode):
        if mode=="train":
            data = self.load_data(os.path.join(self.annotations_path, "train.pkl"))
        elif mode=="test":
            data = self.load_data(os.path.join(self.annotations_path, "test.pkl"))
        rows = []
        for tup in data.iterrows():
            row = tup[1]
            file = row["file"]
            idx = row["index"]
            emg = self.load_data(os.path.join(self.data_path,"raw", file))
            row = emg.iloc[idx].copy()
            row["subject"]=file.split('.')[0]
            rows.append(row)
        return pd.DataFrame(rows)
    
    def save_data(self, path, data):
        with open(path, 'wb') as f:
            pd.to_pickle(data, f)

    def load_data(path):
        with open (path, 'rb') as f:
            df = pd.read_pickle(f)
        return df

    