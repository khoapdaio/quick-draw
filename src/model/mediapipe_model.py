import cv2
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands


class MediaPipeModel:
	def __init__(self):
		self.hands = mp_hands.Hands(
			min_detection_confidence = 0.8,
		    min_tracking_confidence = 0.8
		)
		self.mp_draw = mp.solutions.drawing_utils
		self.finger_tips = [
			mp_hands.HandLandmark.THUMB_TIP,
			mp_hands.HandLandmark.INDEX_FINGER_TIP,
			mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
			mp_hands.HandLandmark.RING_FINGER_TIP,
			mp_hands.HandLandmark.PINKY_TIP,
		]
		self.finger_mcp = [
			mp_hands.HandLandmark.INDEX_FINGER_MCP,
			mp_hands.HandLandmark.INDEX_FINGER_DIP,
			mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
			mp_hands.HandLandmark.RING_FINGER_DIP,
			mp_hands.HandLandmark.PINKY_DIP,
		]

	def draw_landmark(self, frame, hand_landmark):
		self.mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

	def detect_fingers(self, frame):
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		result = self.hands.process(rgb_frame)
		if result.multi_hand_landmarks:
			for hand_landmarks in result.multi_hand_landmarks:
				self.mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

				# Đếm số ngón tay giơ lên
				fingers_up = 0
				for tip, mcp in zip(self.finger_tips, self.finger_mcp):
					if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:  # Nếu đầu ngón tay cao hơn MCP
						fingers_up += 1

				print(f"Fingers up: {fingers_up}")
				return fingers_up, self.__get_pos_of_finger(fingers_up, hand_landmarks, frame.shape[1], frame.shape[0])

		return None, None

	def __get_pos_of_finger(self, fingers_up, hand_landmark, w, h):
		index_finger_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
		middle_finger_tip = hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

		if fingers_up == 1:
			if index_finger_tip.y < middle_finger_tip.y:
				return self.__cal_pos_of_fingers([index_finger_tip], w, h)

			if middle_finger_tip.y < index_finger_tip.y:
				return self.__cal_pos_of_fingers([middle_finger_tip], w, h)

		return self.__cal_pos_of_fingers([index_finger_tip, middle_finger_tip], w, h)

	def __cal_pos_of_fingers(self, list_pos_finger: [], w, h):
		x = 0
		y = 0
		for pos in list_pos_finger:
			x += pos.x
			y += pos.y
		x = int(x / len(list_pos_finger) * w)
		y = int(y / len(list_pos_finger) * h)
		return x, y
