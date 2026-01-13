try:
    from tools import *
except (ImportError, ModuleNotFoundError):
    from lib.tools import *
from torch.utils.data import Dataset


class SensorsDataset(Dataset):
    def __init__(self, numpy_file='..\\datasets\\dataset_1_024s.npy', sensors=[1,2,3], groups=[1], normalize=True, quantize_data=True):
        # Segments, label, group
        if type(sensors) == int:
            sensors = list(sensors)

        if type(numpy_file) == str:
            print(f"Reading {numpy_file} file...", '\r')
            data = np.load(numpy_file)
            print(f"Finished loading {numpy_file} file!", '\r')
        else:
            num = str(numpy_file).replace('.', '_')
            print(f"Reading .\\datasets\\dataset_{num}s.npy file...", end='\r')
            data = np.load(f'.\\datasets\\dataset_{num}s.npy')
            print(f"Finished loading .\\datasets\\dataset_{num}s.npy file!", end='\r')

        if type(groups) == str:
            groups = [groups]
        samples_aux = [np.where(data[:, -1] == group)[0] for group in groups]
        samples = []
        if len(groups) > 1:
            samples = np.concatenate(samples_aux)
        else:
            samples = np.ravel(samples_aux)
        print(np.shape(samples))

        all_data = []
        num_cols = data.shape[1] // 12
        for s in sensors:
            aux = data[samples, (s-1) * num_cols:s * num_cols]
            if len(all_data) == 0:
                all_data = aux
            else:
                all_data = np.hstack([all_data, aux])

        all_data = np.array(all_data, dtype=np.float)
        if quantize_data:
            self.normalized_data = np.array(normalizeFeatures(all_data)[0], dtype=np.float)
            self.quantized_data = np.array(quantize(self.normalized_data), dtype=np.float)
        if normalize and not quantize_data:
            self.normalized_data, _, _ = normalizeFeatures(all_data)
        
        self.labels = data[samples, -2]
        self.groups = data[samples, -1]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.quantized_data[idx], self.labels[idx], self.groups[idx]
