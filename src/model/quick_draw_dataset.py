import os

import numpy as np
from torch.utils.data import Dataset


class QuickDrawDataset(Dataset):
	def __init__(self, data_paths, label_mapping):
		self.data = []
		self.labels = []
		for path in data_paths:
			data = np.load(path)
			print(path)
			self.data.append(data)
			# Ánh xạ nhãn từ file
			file_name = os.path.basename(path)
			name, ext = os.path.splitext(file_name)
			self.labels.extend([label_mapping[name]] * len(data))
		self.data = np.concatenate(self.data, axis = 0)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		image = self.data[idx].reshape(1, 28, 28).astype(np.float32)
		label = self.labels[idx]
		return image, label
