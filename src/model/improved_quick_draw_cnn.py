import torch.nn as nn


# Improved CNN Model with BatchNorm and Dropout
class ImprovedQuickDrawCNN(nn.Module):
	def __init__(self, num_classes):
		super(ImprovedQuickDrawCNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)
		)
		self.classifier = nn.Sequential(
			nn.Linear(128 * 3 * 3, 256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x
