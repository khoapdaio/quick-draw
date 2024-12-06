import cv2
import torch
import yaml

from train import ImprovedQuickDrawCNN


def label_dict_from_config_file(relative_path):
	with open(relative_path, "r") as f:
		label_tag = yaml.full_load(f)["draw"]
	return label_tag


def get_images(path, classes):
	images = [cv2.imread("../{}/{}.png".format(path, item), cv2.IMREAD_UNCHANGED) for item in classes]
	return images


def get_model_predict():
	model = ImprovedQuickDrawCNN(20)
	if torch.cuda.is_available():
		model.load_state_dict(torch.load("../trained_models/improved_quickdraw_model.pth"))
	else:
		model.load_state_dict(
			torch.load("../trained_models/improved_quickdraw_model.pth", map_location = lambda storage, loc: storage))
	model.eval()

	return model
