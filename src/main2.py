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
#OpenCVの設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
#Mediapipeの設定
face_base_options = python.BaseOptions(
	model_asset_path='../models/blaze_face_full_range.tflite'
)
face_options = vision.FaceDetectorOptions(
	base_options=face_base_options,
	running_mode=vision.RunningMode.VIDEO,
	min_detection_confidence=0.5
)
face_detector = vision.FaceDetector.create_from_options(face_options)
hands_base_options = python.BaseOptions(
	model_asset_path='../models/hand_landmarker.task'
)
hands_options = vision.HandLandmarkerOptions(
	base_options=hands_base_options,
	running_mode=vision.RunningMode.VIDEO,
	num_hands=1,
	min_hand_detection_confidence=0.3,
	min_hand_presence_confidence=0.8,
	min_tracking_confidence=0.1
)
hands_landmarker = vision.HandLandmarker.create_from_options(hands_options)
connections = [
	(0, 1), (1, 2), (2, 3), (3, 4),
	(0, 5), (5, 6), (6, 7), (7, 8),
	(5, 9), (9, 10), (10, 11), (11, 12),
	(9, 13), (13, 14), (14, 15), (15, 16),
	(13, 17), (17, 18), (18, 19), (19, 20),
	(0, 17)
]
#顔検出の設定
face_lost_count = 0
face_lost_limit = 10
face_smooth_x = None
face_smooth_y = None
face_smooth_w = None
face_smooth_h = None
face_alpha = 0.1
face_ratio_x = 1.2
face_ratio_y = 0.8
offset_x = 0
offset_y = 0
#pyautoguiに関する設定
pyautogui.PAUSE = 0
cursor_alpha = 0.3
threshold = 1
margin_x = 0.2
margin_y = 0.2
min_x = margin_x
max_x = 1 - margin_x
min_y = margin_y
max_y = 1 - margin_y
cursor_x = None
cursor_y = None
smooth_x = None
smooth_y = None
last_click_time = 0
click_cooldown = 0.3
#ジェスチャーの設定
thumb_angle = None
first_angle = None
second_angle = None
third_angle = None
fourth_angle = None
gesture = None
open_palm_line = 120
#描画の設定
circle_radius = 8
circle_color = (0, 255, 0)
line_thickness = 4
#EMA関数
def ema(prev_x, prev_y, target_x, target_y, cursor_alpha):
	x = prev_x + cursor_alpha * (target_x - prev_x)
	y = prev_y + cursor_alpha * (target_y - prev_y)
	return x, y
#デッドゾーン関数
def in_deadzone(prev_x, prev_y, x, y, threshold):
	distance = math.hypot(x - prev_x, y - prev_y)
	return distance < threshold
#angle関数
def angle(a, b, c):
	ab_x = a.x - b.x
	ab_y = a.y - b.y
	cb_x = c.x - b.x
	cb_y = c.y - b.y
	dot = ab_x * cb_x + ab_y * cb_y

	ab_len = math.hypot(ab_x, ab_y)
	cb_len = math.hypot(cb_x, cb_y)
	cos_angle = dot / (ab_len * cb_len)
	cos_angle = max(-1.0, min(1.0, cos_angle))
	return math.degrees(math.acos(cos_angle))
#angle_3d関数
def angle_3d(a, b, c):
	ab_x = a.x - b.x
	ab_y = a.y - b.y
	ab_z = a.z - b.z
	cb_x = c.x - b.x
	cb_y = c.y - b.y
	cb_z = c.z - b.z
	dot = (
		ab_x * cb_x
		+ ab_y * cb_y
		+ ab_z * cb_z
	)
	ab_len = math.sqrt(
		ab_x**2
		+ ab_y**2
		+ ab_z**2
	)
	cb_len = math.sqrt(
		cb_x**2
		+ cb_y**2
		+ cb_z**2
	)
	cos_angle = dot / (ab_len * cb_len)
	cos_angle = max(-1.0, min(1.0, cos_angle))
	return math.degrees(math.acos(cos_angle))
#gesture関数
def gesture_judge(first_angle, second_angle, third_angle, fourth_angle):
	if (
		first_angle > open_palm_line
		and second_angle > open_palm_line
		and third_angle > open_palm_line
		and fourth_angle > open_palm_line
	):
		gesture = 'Opened Palm'
	elif (
		first_angle <= open_palm_line
		and second_angle <= open_palm_line
		and third_angle <= open_palm_line
		and fourth_angle <= open_palm_line
	):
		gesture = 'Closed Palm'
	elif (
		first_angle > open_palm_line
		and second_angle > open_palm_line
		and third_angle <= open_palm_line
		and third_angle <= open_palm_line
	):
		gesture = 'Victory'
	else:
		gesture = 'Others'
	print(gesture)
	return gesture

#カメラが読み込まれてる時にループ
while cap.isOpened():
	ret, frame = cap.read()
	if ret == False:
		break
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	mp_image = mp.Image(
		image_format=mp.ImageFormat.SRGB,
		data=rgb
	)
	timestamp_ms = time.monotonic_ns() // 1_000_000
	hands_result = hands_landmarker.detect_for_video(
		mp_image,
		timestamp_ms
	)
	face_result = face_detector.detect_for_video(
		mp_image,
		timestamp_ms
	)
	frame_h, frame_w, _ = frame.shape
	#顔検出
	if face_result.detections:
		face_lost_count = 0
		detection = face_result.detections[0]
		bbox = detection.bounding_box
		face_x = bbox.origin_x / frame_w
		face_y = bbox.origin_y / frame_h
		face_w = bbox.width / frame_w
		face_h = bbox.height / frame_h
		#最初だけ
		if face_smooth_x == None:
			face_smooth_x = face_x
			face_smooth_y = face_y
			face_smooth_w = face_w
			face_smooth_h = face_h
		# 顔の位置・大きさを平滑化
		face_smooth_x, face_smooth_y = ema(
			face_smooth_x,
			face_smooth_y,
			face_x,
			face_y,
			face_alpha
		)
		face_smooth_w, face_smooth_h = ema(
			face_smooth_w,
			face_smooth_h,
			face_w,
			face_h,
			face_alpha
		)
		face_center_x = face_smooth_x + face_smooth_w / 2
		face_center_y = face_smooth_y + face_smooth_h / 2
		offset_y = face_smooth_h * 1.4
		min_x = face_center_x - face_smooth_w * face_ratio_x
		max_x = face_center_x + face_smooth_w * face_ratio_x
		min_y = face_center_y - face_smooth_h * face_ratio_y + offset_y
		max_y = face_center_y + face_smooth_h * face_ratio_y + offset_y
		min_x = max(0, min_x)
		max_x = min(1, max_x)
		min_y = max(0, min_y)
		max_y = min(1, max_y)
	else:
		face_lost_count += 1
		if face_lost_count > face_lost_limit:
			#固定された範囲
			min_x = margin_x
			max_x = 1 - margin_x
			min_y = margin_y
			max_y = 1 - margin_y

	#検知する範囲の描画
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
	if hands_result.hand_landmarks:
		for i, hand_landmarks in enumerate(hands_result.hand_landmarks):
			handedness = hands_result.handedness[i][0]
			handedness_name = handedness.category_name
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

			#ジェスチャーとクリック動作
			first_angle = angle_3d(
				hand_landmarks[5],
				hand_landmarks[6],
				hand_landmarks[8]
			)
			second_angle = angle_3d(
				hand_landmarks[9],
				hand_landmarks[10],
				hand_landmarks[12]
			)
			third_angle = angle_3d(
				hand_landmarks[13],
				hand_landmarks[14],
				hand_landmarks[16]
			)
			fourth_angle = angle_3d(
				hand_landmarks[17],
				hand_landmarks[18],
				hand_landmarks[20]
			)
			gesture = gesture_judge(
				first_angle, 
				second_angle, 
				third_angle, 
				fourth_angle
			)
			if gesture == 'Closed Palm':
				current_time = time.monotonic()
				if current_time - last_click_time >= click_cooldown:
					pyautogui.click()
					last_click_time = current_time
			if gesture == 'Victory':
				current_time = time.monotonic()
				if current_time - last_click_time >= click_cooldown:
					pyautogui.rightClick()
					last_click_time = current_time
			#マウス操作
			if gesture == 'Opened Palm' or gesture == 'Others':
				mp_palm_x = max(min_x, min(max_x, mp_palm_x))
				mp_palm_y = max(min_y, min(max_y, mp_palm_y))
				target_x = int((1 - (mp_palm_x - min_x) / (max_x - min_x)) * screen_w)
				target_y = int((mp_palm_y - min_y) / (max_y - min_y) * screen_h)
				#初回のみ
				if smooth_x == None:
					smooth_x = target_x
					smooth_y = target_y
					cursor_x = target_x
					cursor_y = target_y
				#EMA関数を使用して滑らかに
				smooth_x, smooth_y = ema(smooth_x, smooth_y, target_x, target_y, cursor_alpha)
				#デッドゾーン
				if in_deadzone(cursor_x, cursor_y, smooth_x, smooth_y, threshold) == False:
					cursor_x = smooth_x
					cursor_y = smooth_y
					pyautogui.moveTo(cursor_x, cursor_y)
	
	#確認用ウインドウの表示
	frame = cv2.flip(frame, 1)
	display = cv2.resize(frame, (int(screen_w / 2), int(screen_h /2)))
	cv2.imshow('Hand Tracking', display)
	#Qキーを押すとループから抜ける
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
#確認用ウインドウの削除
cap.release()
cv2.destroyAllWindows()