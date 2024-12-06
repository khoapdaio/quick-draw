# Xử lý phần nhận hình ảnh
import cv2


class CameraModel:

	def __init__(self, h = 480, w = 640):
		self.cap = cv2.VideoCapture(0)
		self.h, self.w = h, w
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
		self.cap.set(cv2.CAP_PROP_FPS, 60)

	def get_frame(self):
		ret, frame = self.cap.read()
		frame = cv2.flip(frame, 1)
		return cv2.resize(frame, (self.w, self.h))

	def get_size(self):
		return self.h, self.w

	def shutdown(self):
		self.cap.release()
		cv2.destroyAllWindows()

	def show(self, frame, canvases: [], alpha = 0.8, beta = 1, gamma = 0):
		combined_frame = frame
		for canvas in canvases:
			# Chồng canvas lên khung hình webcam
			combined_frame = cv2.addWeighted(combined_frame, alpha, canvas, beta, gamma)

		# Hiển thị khung hình với canvas
		cv2.imshow("Camera with Drawing", combined_frame)
