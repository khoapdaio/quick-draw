import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model.improved_quick_draw_cnn import ImprovedQuickDrawCNN
from model.quick_draw_dataset import QuickDrawDataset

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")


def label_dict_from_config_file(relative_path):
	with open(relative_path, "r") as f:
		label_tag = yaml.full_load(f)["draw"]
	return label_tag


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
	model.train()
	for epoch in range(num_epochs):
		epoch_loss = 0.0
		correct = 0
		total = 0
		loop = tqdm(train_loader, leave = True, desc = f"Epoch [{epoch + 1}/{num_epochs}]")
		accuracy = 0.0

		for images, labels in loop:
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()

			outputs = model(images)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

			accuracy = 100.0 * correct / total
			loop.set_postfix(loss = loss.item(), acc = accuracy)

		scheduler.step()
		logging.info(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {epoch_loss:.4f}, Train Accuracy = {accuracy:.2f}%")

		# Validation phase
		val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
		logging.info(f"Epoch {epoch + 1}/{num_epochs}: Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.2f}%")


# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
	model.eval()
	val_loss = 0.0
	correct = 0
	total = 0
	loop = tqdm(train_loader, leave = True, desc = f"Epoch []")
	with torch.no_grad():
		for images, labels in loop:
			images, labels = images.to(device), labels.to(device)

			# Forward pass
			outputs = model(images)
			loss = criterion(outputs, labels)
			val_loss += loss.item()

			# Metrics
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()
			accuracy = 100.0 * correct / total
			loop.set_postfix(loss = loss.item(), acc = accuracy)

	val_loss /= len(dataloader)
	accuracy = 100.0 * correct / total
	return val_loss, accuracy


# Main script
if __name__ == "__main__":
	# Configurations
	data_dir = "../quickdraw_data"  # Update this path to your dataset directory
	batch_size = 64
	num_classes = 20  # Number of classes in QuickDraw dataset
	num_epochs = 5
	learning_rate = 0.001
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	list_label = label_dict_from_config_file("../quick_draw.yaml")
	# Load data
	print("Loading data...")
	data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]
	dataset = QuickDrawDataset(data_paths, label_mapping = list_label)

	# Split dataset into train, val, and test
	train_size = int(0.7 * len(dataset))
	val_size = int(0.2 * len(dataset))
	test_size = len(dataset) - train_size - val_size
	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
	test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

	logging.info(f"Dataset split: Train = {len(train_dataset)}, Val = {len(val_dataset)}, Test = {len(test_dataset)}")

	# Initialize model, loss, optimizer, and scheduler
	print("Initializing model...")
	model = ImprovedQuickDrawCNN(num_classes = num_classes).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = 1e-4)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

	# Train the model
	logging.info("Starting training...")
	train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs)

	# Evaluate on test set
	logging.info("Evaluating on test set...")
	test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
	logging.info(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.2f}%")

	# Save the model
	torch.save(model.state_dict(), "../trained_models/improved_quickdraw_model.pth")
	print("Model saved successfully!")
