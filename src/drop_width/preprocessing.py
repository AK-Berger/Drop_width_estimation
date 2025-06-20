import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DropDataProcessor:
    def __init__(self, df):
        self.df = df

        # These will be populated after extract_data_label() is called
        self.train_key = None
        self.test_key = None
        self.train_key_set = None
        self.test_key_set = None

    def normalize(self, train_data_input, test_data_input):
        scaler = MinMaxScaler()
        train_data = train_data_input.copy()
        test_data = test_data_input.copy()

        for column in train_data.columns:
            if column in test_data.columns:
                train_data[[column]] = scaler.fit_transform(train_data[[column]])
                test_data[[column]] = scaler.transform(test_data[[column]])
            else:
                print(f"Warning: Column '{column}' not found in both datasets. Skipping normalization.")

        return train_data, test_data

    def extract_data_label(self):
        features = ["Advancing (degree)", "Receding (degree)", "Drop length (cm)",
                    "Drop height (cm)", "Velocity (cm/s)", "Middle line angle (degree)"]
        data = self.df[features]
        if "Drop width (cm)" in self.df.columns:
            label = self.df["Drop width (cm)"]
        else:
            label = [0]*len(self.df)
            label = pd.Series(label)
        key = self.df["Video ID"]
        status = self.df["status"]

        

        # Save keys for slicing later
        self.train_key = key[status == "train"]
        self.test_key = key[status == "test"]
        self.train_key_set = self.train_key.unique()
        self.test_key_set = self.test_key.unique()

        train_data = data[status == "train"]
        test_data = data[status == "test"]
        train_label = label[status == "train"]
        test_label = label[status == "test"]

        train_data_notnormalized = train_data.copy()
        train_data, test_data = self.normalize(train_data, test_data)

        # Scale labels from cm to µm
        CM_TO_UM = 10000
        train_label *= CM_TO_UM
        test_label *= CM_TO_UM

        return train_data, test_data, train_label, test_label, train_data_notnormalized

    def slicing(self, data, window_size):
        data = np.array(data)
        num_slices = data.shape[0] - window_size + 1
        result = np.zeros((num_slices, window_size, data.shape[1]))

        for i in range(num_slices):
            result[i, :, :] = data[i : i + window_size, :]

        return result

    def slicing_label(self, label, window_size):
        label = np.array(label).reshape(-1, 1)
        num_slices = label.shape[0] - window_size + 1
        result = np.zeros((num_slices, 1, 1))

        for i in range(num_slices):
            mid_idx = i + window_size // 2
            result[i, 0, 0] = label[mid_idx, 0]

        return result

    def slice_patch(self, slide_window, train_data, train_label):
        train_data_sliced = np.empty((0, slide_window, train_data.shape[1]))
        train_label_sliced = np.empty((0, 1, 1))

        for key in self.train_key_set:
            mask = self.train_key == key
            temp_data = train_data[mask]
            temp_label = train_label[mask]

            if len(temp_data) >= slide_window:
                data_sliced = self.slicing(temp_data, slide_window)
                label_sliced = self.slicing_label(temp_label, slide_window)

                train_data_sliced = np.concatenate((train_data_sliced, data_sliced), axis=0)
                train_label_sliced = np.concatenate((train_label_sliced, label_sliced), axis=0)

        return train_data_sliced, train_label_sliced

