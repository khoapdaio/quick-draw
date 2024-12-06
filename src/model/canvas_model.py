# xử lý phần vẽ hình ảnh bao gồm các phần
# Phần một vẽ ô start ở góc phải bên trên ô
# Phần 2 thực hiện vẽ dựa trên tọa độ x, y của ngón tay truyền vào,
# Phần 3 thực hiện lấy cạnh hình chữ nhật bao quanh của hình ảnh đã vẽ và trả về định dạng 28,28
import cv2
import numpy as np


class CanvasModel:
	def __init__(self, h = 480, w = 640):
		self.h, self.w = h, w
		self.canvas = np.zeros((self.h, self.w, 3), dtype = np.uint8)
		self.last_position = None
		self.tool_canvas = np.zeros((self.h, self.w, 3), dtype = np.uint8)
		self.start_zone = ((520, 20), (620, 120))

	def draw_starting_frame(self, start = (520, 20), stop = (620, 120)):
		self.start_zone = (start, stop)
		cv2.rectangle(self.tool_canvas, start, stop, (0, 255, 0), 2)
		cv2.putText(self.tool_canvas, "START", (start[0] + 10, stop[1] + 30),
		            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	def draw_on_canvas(self, x, y):
		cv2.line(self.canvas, self.last_position, (x, y), (255, 255, 255), thickness = 5)
		self.last_position = (x, y)

	def save_canvas(self):
		gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

		# Find contours
		contours, _ = cv2.findContours(gray_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			# Get bounding box of the largest contour
			x, y, w, h = cv2.boundingRect(max(contours, key = cv2.contourArea))
			cropped_canvas = gray_canvas[y - 10:y + h + 10, x - 10:x + w + 10]
		else:
			cropped_canvas = gray_canvas

		resized_canvas = cv2.resize(cropped_canvas, (28, 28))
		cv2.imwrite("../drawn/photo.png", resized_canvas)
		print("Hình vẽ đã được lưu thành 'photo.png'")
		return resized_canvas

	def clear_canvas(self):
		self.canvas = np.zeros((self.h, self.w, 3), dtype = np.uint8)

	def clear_tool_canvas(self):
		self.tool_canvas = np.zeros((self.h, self.w, 3), dtype = np.uint8)

	def set_last_position(self, x, y):
		self.last_position = (x, y)

	def get_canvases(self):
		return [self.canvas, self.tool_canvas]

	def is_in_start_area(self, position):
		if position:
			x, y = position[0], position[1]
			start_x, start_y = self.start_zone[0][0], self.start_zone[0][1]
			stop_x, stop_y = self.start_zone[1][0], self.start_zone[1][1]
			return start_x <= x <= stop_x and start_y <= y <= stop_y
		return False

	def draw_predict(self, image):
		x_offset = (self.w - 60) // 2  # Tâm ngang
		y_offset = (self.h - 60) // 5  # Tâm dọc

		image = cv2.resize(image, (60, 60), interpolation = cv2.INTER_CUBIC)
		bg_image = self.tool_canvas[y_offset:y_offset + 60, x_offset:x_offset + 60]
		fg_mask = image[:, :, 3:]
		fg_image = image[:, :, :3]
		bg_mask = 255 - fg_mask
		bg_image = bg_image / 255
		fg_image = fg_image / 255
		fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) / 255
		bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR) / 255
		cv2.putText(
			self.tool_canvas,
			'You are drawing',
			(100, 50),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.5,
			(0, 255, 0),
			5,
			cv2.LINE_AA
		)
		blended = cv2.addWeighted(bg_image * bg_mask, 255, fg_image * fg_mask, 255, 0.).astype(np.uint8)
		self.tool_canvas[y_offset:y_offset + 60, x_offset:x_offset + 60] = blended
