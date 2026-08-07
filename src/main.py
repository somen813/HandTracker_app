from screeninfo import get_monitors
import pyautogui
import cv2
import mediapipe as mp

pyautogui.PAUSE = 0
monitors = get_monitors()
alpha = 0.5
prev_x = 0
prev_y = 0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, monitors[0].width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, monitors[0].height)
cap.set(cv2.CAP_PROP_FPS, 30)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
	static_image_mode=False,
	max_num_hands=2,
	min_detection_confidence=0.8,
	min_tracking_confidence=0.5
)

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break
		
	cap_flip = cv2.flip(frame, 1)
	rgb = cv2.cvtColor(cap_flip, cv2.COLOR_BGR2RGB)
	results = hands.process(rgb)

	if results.multi_hand_landmarks:
		for hand_landmarks, handedness in zip(
			results.multi_hand_landmarks,
			results.multi_handedness
		):
			label = handedness.classification[0].label
			score = handedness.classification[0].score
			if label == 'Right':
				line_color = (0, 0, 255)
			else:
				line_color = (255, 0, 0)
			circle_color = (0, 255, 0)

			h, w, _ = frame.shape
			for start, end in mp_hands.HAND_CONNECTIONS:
				p1 = hand_landmarks.landmark[start]
				p2 = hand_landmarks.landmark[end]
				x1 = int(p1.x * w)
				y1 = int(p1.y * h)
				x2 = int(p2.x * w)
				y2 = int(p2.y * h)
				cv2.line(cap_flip, (x1, y1), (x2, y2), line_color, 3)

			for lm in hand_landmarks.landmark:
				x = int(lm.x * w)
				y = int(lm.y * h)
				cv2.circle(cap_flip, (x, y), 8, circle_color, -1)

			index_tip = hand_landmarks.landmark[
				mp_hands.HandLandmark.INDEX_FINGER_TIP
			]
			x_tip = index_tip.x
			y_tip = index_tip.y
			smooth_x = prev_x + alpha * (x_tip - prev_x)
			smooth_y = prev_y + alpha * (y_tip - prev_y)
				
			if abs(x_tip - smooth_x) > 5 or abs(y_tip - smooth_y) > 2:
				pyautogui.moveTo(smooth_x * monitors[0].width, smooth_y * monitors[0].height)

			prev_x = x_tip
			prev_y = y_tip

	display = cv2.resize(cap_flip, None, fx=0.5, fy=0.5)
	cv2.imshow('Hand Tracking', display)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWidows()