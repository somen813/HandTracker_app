#importするもの
import cv2
import mediapipe as mp
import math
from screeninfo import get_monitors
import pyautogui
from statistics import mean

#モニターの情報取得
monitors = get_monitors()
screen_w = monitors[0].width
screen_h = monitors[0].height
#OpenCvの設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_w/2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_h/2)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)
#Mediapipeの設定
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
	static_image_mode=False,
	max_num_hands=1,
	min_detection_confidence=0.7,
	min_tracking_confidence=0.6
)
#pyautoguiに関する設定
pyautogui.PAUSE = 0
alpha = 0.3
threshold = 4
cursor_x = None
cursor_y = None
smooth_x = None
smooth_y = None
#描画の設定
circle_radius = 4
circle_color = (0, 255, 0)
line_thickness = 2
#EMA関数
def ema(prev_x, prev_y, target_x, target_y, alpha):
	x = prev_x + alpha * (target_x - prev_x)
	y = prev_y + alpha * (target_y - prev_y)
	return x, y
#デッドゾーン関数
def in_deadzone(prev_x, prev_y, x, y, threshold):
	distance = math.hypot(x - prev_x, y - prev_y)
	return distance < threshold

#カメラが読み込まれてる時にループ
while cap.isOpened():
	ret, frame = cap.read()
	if ret == False:
		break

	frame = cv2.flip(frame, 1)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = hands.process(rgb)

	if results.multi_hand_landmarks:
		for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
			#右手 -> 赤, 左手 -> 青
			label = handedness.classification[0].label
			if label == 'Right':
				line_color = (0, 0, 255)
			else:
				line_color = (255, 0, 0)
			#点と線の色指定
			frame_h, frame_w, _ = frame.shape
			for start, end in mp_hands.HAND_CONNECTIONS:
				mp_p1 = hand_landmarks.landmark[start]
				mp_p2 = hand_landmarks.landmark[end]
				frame_x1 = int(mp_p1.x * frame_w)
				frame_y1 = int(mp_p1.y * frame_h)
				frame_x2 = int(mp_p2.x * frame_w)
				frame_y2 = int(mp_p2.y * frame_h)
				cv2.line(frame, (frame_x1, frame_y1), (frame_x2, frame_y2), line_color, line_thickness)
			for lm in hand_landmarks.landmark:
				frame_x = int(lm.x * frame_w)
				frame_y = int(lm.y * frame_h)
				cv2.circle(frame, (frame_x, frame_y), circle_radius, circle_color, -1)
			indices = [0, 1, 5, 9, 13, 17]
			mp_palm_x = mean([hand_landmarks.landmark[i].x for i in indices])
			mp_palm_y = mean([hand_landmarks.landmark[i].y for i in indices])
			frame_palm_x = int(mp_palm_x * frame_w)
			frame_palm_y = int(mp_palm_y * frame_h)
			cv2.circle(frame, (frame_palm_x, frame_palm_y), circle_radius, (0, 255, 255), -1)

			#マウス操作
			target_x = int(mp_palm_x * screen_w)
			target_y = int(mp_palm_y * screen_h)
			#初回のみ
			if smooth_x is None:
				smooth_x = target_x
				smooth_y = target_y
				cursor_x = target_x
				cursor_y = target_y
			smooth_x, smooth_y = ema(smooth_x, smooth_y, target_x, target_y, alpha)
			#デッドゾーン
			if in_deadzone(cursor_x, cursor_y, smooth_x, smooth_y, threshold) == False:
				cursor_x = smooth_x
				cursor_y = smooth_y
				pyautogui.moveTo(cursor_x, cursor_y)

	#確認用ウインドウの表示
	cv2.imshow('Hand Tracking', frame)
	#Qキーを押すとループから抜ける
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#確認用ウインドウの削除
cap.release()
cv2.destroyAllWindows()