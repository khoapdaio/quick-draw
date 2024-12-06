# chứa phần code xử lý logic chính
import cv2
import numpy as np
import torch

from util.config import *
from model.camera_model import CameraModel
from model.canvas_model import CanvasModel
from model.mediapipe_model import MediaPipeModel
from util.file_util import get_images, get_model_predict


def main():
	camera_model = CameraModel()
	canvas_model = CanvasModel()
	mediapipe_model = MediaPipeModel()
	canvas_model.draw_starting_frame()
	model = get_model_predict()
	class_images = get_images("images", CLASSES)
	process = 0
	is_drawing = False
	start_process = 0
	while True:
		frame = camera_model.get_frame()
		fingers_up, position = mediapipe_model.detect_fingers(frame)

		if canvas_model.is_in_start_area(position):
			start_process += 1
			if start_process == 50:
				is_drawing = True
				canvas_model.clear_tool_canvas()
				start_process = 0
		else:
			start_process = 0

		if fingers_up == 2:
			process = 0
			cv2.circle(frame, (position[0], position[1]), 8, (0, 255, 0), -1)  # Vòng tròn xanh
			canvas_model.set_last_position(position[0], position[1])

		elif fingers_up == 1:
			process = 0
			cv2.circle(frame, (position[0], position[1]), 8, (0, 0, 255), -1)  # Vòng tròn đỏ
			if is_drawing:
				canvas_model.draw_on_canvas(position[0], position[1])

		elif fingers_up == 5:
			canvas_model.clear_canvas()
			canvas_model.set_last_position(position[0], position[1])

		elif fingers_up == 0:
			process += 1
			if process == 50:
				image = canvas_model.save_canvas()
				image = np.array(image, dtype = np.float32)[None, None, :, :]
				image = torch.tensor(image, dtype = torch.float32)
				predict = model(image)
				canvas_model.clear_canvas()
				predicted_class = torch.argmax(predict, dim = 1)
				print(f"Predicted class: {CLASSES[predicted_class.item()]}")
				is_drawing = False
				canvas_model.draw_starting_frame()
				canvas_model.draw_predict(class_images[predicted_class.item()])

		camera_model.show(frame, canvas_model.get_canvases())
		if cv2.waitKey(1) & 0xFF == 27:
			break
	camera_model.shutdown()


# Chạy ứng dụng
if __name__ == "__main__":
	main()
