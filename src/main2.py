#importするもの
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import pyautogui
from screeninfo import get_monitors
from statistics import mean
import time

#モニターの情報取得
monitors = get_monitors()
screen_w = monitors[0].width
screen_h = monitors[0].height
#OpenCvの設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
#Mediapipeの設定
base_options = python .BaseOptions(
	model_asset_path='../models/hand_landmarker.task'
)
options = vision.HandLandmarkerOptions(
	base_options=base_options,
	running_mode=vision.RunningMode.VIDEO,
	num_hands=1,
	min_hand_detection_confidence=0.3,
	min_hand_presence_confidence=0.8,
	min_tracking_confidence=0.1
)
detector = vision.HandLandmarker.create_from_options(options)
timestamp_ms = 0
connections = [
	(0, 1), (1, 2), (2, 3), (3, 4),
	(0, 5), (5, 6), (6, 7), (7, 8),
	(5, 9), (9, 10), (10, 11), (11, 12),
	(9, 13), (13, 14), (14, 15), (15, 16),
	(13, 17), (17, 18), (18, 19), (19, 20),
	(0, 17)
	]
#pyautoguiに関する設定
pyautogui.PAUSE = 0
alpha = 0.3
threshold = 3
margin_x = 0.2
margin_y = 0.2
min_x = margin_x
max_x = 1- margin_x
min_y = margin_y
max_y = 1- margin_y
cursor_x = None
cursor_y = None
smooth_x = None
smooth_y = None
#描画の設定
circle_radius = 8
circle_color = (0, 255, 0)
line_thickness = 4
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
	mp_image = mp.Image(
		image_format=mp.ImageFormat.SRGB,
		data=rgb
	)
	timestamp_ms = time.monotonic_ns() // 1_000_000
	result = detector.detect_for_video(
		mp_image,
		timestamp_ms
	)

	#検知する範囲の描画
	frame_h, frame_w, _ = frame.shape
	left = int(frame_w * min_x)
	right = int(frame_w * max_x)
	top = int(frame_h * min_y)
	bottom = int(frame_h * max_y)
	overlay = frame.copy()
	cv2.rectangle(overlay, (0,0), (frame_w, frame_h), (0,0,0), -1)
	overlay[top:bottom, left:right] = frame[top:bottom, left:right]
	frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 4)

	#手のランドマークの描画、及びカーソル操作
	if result.hand_landmarks:
		for i, hand_landmarks in enumerate(result.hand_landmarks):
			handedness = result.handedness[i][0]
			handedness_name = handedness.category_name
			#ウインドウを反転してるので反転処理
			if handedness_name == 'Right':
				handedness_name = 'Left'
			elif handedness_name == 'Left':
				handedness_name = 'Right'
			#右手 -> 赤, 左手 -> 青
			if handedness_name == 'Right':
				line_color = (0, 0, 255)
			else:
				line_color = (255, 0, 0)
			#点と線の描画
			for start, end in connections:
				mp_p1 = hand_landmarks[start]
				mp_p2 = hand_landmarks[end]
				frame_x1 = int(mp_p1.x * frame_w)
				frame_y1 = int(mp_p1.y * frame_h)
				frame_x2 = int(mp_p2.x * frame_w)
				frame_y2 = int(mp_p2.y * frame_h)
				cv2.line(frame, (frame_x1 ,frame_y1), (frame_x2, frame_y2), line_color, 2)
			for landmark in hand_landmarks:
				frame_x = int(landmark.x * frame_w)
				frame_y = int(landmark.y * frame_h)
				cv2.circle(frame, (frame_x, frame_y), 5, (0, 255, 0), -1)

			#手のひらの重心に黄色の点
			indices = [0, 1, 5, 9, 13, 17]
			mp_palm_x = mean([hand_landmarks[i].x for i in indices])
			mp_palm_y = mean([hand_landmarks[i].y for i in indices])
			frame_palm_x = int(mp_palm_x * frame_w)
			frame_palm_y = int(mp_palm_y * frame_h)
			cv2.circle(frame, (frame_palm_x, frame_palm_y), circle_radius, (0, 255, 255), -1)
			#マウス操作
			mp_palm_x = max(min_x, min(max_x, mp_palm_x))
			mp_palm_y = max(min_y, min(max_y, mp_palm_y))
			target_x = int((mp_palm_x - min_x) / (max_x - min_x) * screen_w)
			target_y = int((mp_palm_y - min_y) / (max_y - min_y) * screen_h)
			#初回のみ
			if smooth_x is None:
				smooth_x = target_x
				smooth_y = target_y
				cursor_x = target_x
				cursor_y = target_y
			#EMA関数を使用して滑らかに
			smooth_x, smooth_y = ema(smooth_x, smooth_y, target_x, target_y, alpha)
			#デッドゾーン
			if in_deadzone(cursor_x, cursor_y, smooth_x, smooth_y, threshold) == False:
				cursor_x = smooth_x
				cursor_y = smooth_y
				pyautogui.moveTo(cursor_x, cursor_y)

	#確認用ウインドウの表示
	display = cv2.resize(frame, (int(screen_w / 2), int(screen_h /2)))
	cv2.imshow('Hand Tracking', display)
	#Qキーを押すとループから抜ける
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#確認用ウインドウの削除
cap.release()
cv2.destroyAllWindows()