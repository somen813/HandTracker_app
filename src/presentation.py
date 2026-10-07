import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import pyautogui
from screeninfo import get_monitors
from statistics import mean
import time
#====================
#設定
#====================

#カメラ
CAMERA_INDEX = 0
CAMERA_WIDTH = 1980
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
#顔検出
FACE_MODEL_PATH = '../models/blaze_face_full_range.tflite'
FACE_MIN_DETECTION_CONFIDENCE = 0.5

FACE_LOST_LIMIT = 10
FACE_SMOOTH_ALPHA = 0.1
FACE_RATIO_X = 1.2
FACE_RATIO_Y = 0.8
FACE_OFFSET_X = 0
FACE_OFFSET_Y = 0
#手検出
HAND_MODEL_PATH = '../models/hand_landmarker.task'
HAND_NUM = 1
HAND_MIN_DETECTION_CONFIDENCE = 0.3
HAND_MIN_HAND_PRESENCE_CONFIDENCE = 0.8
HAND_MIN_TRACKING_CONFIDENCE = 0.1
HAND_CONNECTIONS = [
	(0, 1), (1, 2), (2, 3), (3, 4),
	(0, 5), (5, 6), (6, 7), (7, 8),
	(5, 9), (9, 10), (10, 11), (11, 12),
	(9, 13), (13, 14), (14, 15), (15, 16),
	(13, 17), (17, 18), (18, 19), (19, 20),
	(0, 17)
]
#カーソル
CURSOR_ALPHA = 0.3
CURSOR_THRESHOLD = 1
CLICK_COOLDOWN = 0.3
#ジェスチャー
OPEN_PALM_AGNLE_THRESHOLD = 120
#描画
PALM_CIRCLE_RADIUS = 8
PALM_CIRCLE_COLOR = (0, 255, 0)
PALM_LINE_THICKNESS = 4

#====================
#初期化
#====================

face_lost_count = 0
face_smooth_x_norm = None
face_smooth_y_norm = None
face_smooth_width_norm = None
face_smooth_height_norm = None

pyautogui.PAUSE = 0
margin_x = 0.2
margin_y = 0.2
cursor_min_x_norm = margin_x
cursor_max_x_norm = 1 - margin_x
cursor_min_y_norm = margin_y
cursor_max_y_norm = 1 - margin_y
cursor_x_px = None
cursor_y_px = None
cursor_smooth_x_px = None
cursor_smooth_y_px = None
last_click_time = 0

thumb_angle = None
index_finger_angle = None
middle_finger_angle = None
ring_finger_angle = None
pinky_finger_angle = None
gesture = None

#モニターの情報取得
monitors = get_monitors()
screen_width = monitors[0].width
screen_height = monitors[0].height
#OpenCVの設定
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
#顔検出
face_base_options = python.BaseOptions(
	model_asset_path=FACE_MODEL_PATH
)
face_options = vision.FaceDetectorOptions(
	base_options=face_base_options,
	running_mode=vision.RunningMode.VIDEO,
	min_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE
)
face_detector = vision.FaceDetector.create_from_options(face_options)
#手検出
hands_base_options = python.BaseOptions(
	model_asset_path=HAND_MODEL_PATH
)
hands_options = vision.HandLandmarkerOptions(
	base_options=hands_base_options,
	running_mode=vision.RunningMode.VIDEO,
	num_hands=1,
	min_hand_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
	min_hand_presence_confidence=HAND_MIN_HAND_PRESENCE_CONFIDENCE,
	min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE
)
hands_landmarker = vision.HandLandmarker.create_from_options(hands_options)

#関数
#EMA関数
def ema(prev_x, prev_y, target_x_px, target_y_px, cursor_alpha):
	x = prev_x + cursor_alpha * (target_x_px - prev_x)
	y = prev_y + cursor_alpha * (target_y_px - prev_y)
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
def gesture_judge(index_finger_angle, middle_finger_angle, ring_finger_angle, pinky_finger_angle):
	if (
		index_finger_angle > OPEN_PALM_AGNLE_THRESHOLD
		and middle_finger_angle > OPEN_PALM_AGNLE_THRESHOLD
		and ring_finger_angle > OPEN_PALM_AGNLE_THRESHOLD
		and pinky_finger_angle > OPEN_PALM_AGNLE_THRESHOLD
	):
		gesture = 'Opened Palm'
	elif (
		index_finger_angle <= OPEN_PALM_AGNLE_THRESHOLD
		and middle_finger_angle <= OPEN_PALM_AGNLE_THRESHOLD
		and ring_finger_angle <= OPEN_PALM_AGNLE_THRESHOLD
		and pinky_finger_angle <= OPEN_PALM_AGNLE_THRESHOLD
	):
		gesture = 'Closed Palm'
	elif (
		index_finger_angle > OPEN_PALM_AGNLE_THRESHOLD
		and middle_finger_angle > OPEN_PALM_AGNLE_THRESHOLD
		and ring_finger_angle <= OPEN_PALM_AGNLE_THRESHOLD
		and pinky_finger_angle <= OPEN_PALM_AGNLE_THRESHOLD
	):
		gesture = 'Victory'
	else:
		gesture = 'Others'
	print(gesture)
	return gesture

#メインループ
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
	frame_height, frame_width, _ = frame.shape
	#顔検出
	if face_result.detections:
		face_lost_count = 0
		detection = face_result.detections[0]
		bbox = detection.bounding_box
		face_x_norm = bbox.origin_x / frame_width
		face_y_norm = bbox.origin_y / frame_height
		face_width_norm = bbox.width / frame_width
		face_height_norm = bbox.height / frame_height
		#最初だけ
		if face_smooth_x_norm == None:
			face_smooth_x_norm = face_x_norm
			face_smooth_y_norm = face_y_norm
			face_smooth_width_norm = face_width_norm
			face_smooth_height_norm = face_height_norm
		# 顔の位置・大きさを平滑化
		face_smooth_x_norm, face_smooth_y_norm = ema(
			face_smooth_x_norm,
			face_smooth_y_norm,
			face_x_norm,
			face_y_norm,
			FACE_SMOOTH_ALPHA
		)
		face_smooth_width_norm, face_smooth_height_norm = ema(
			face_smooth_width_norm,
			face_smooth_height_norm,
			face_width_norm,
			face_height_norm,
			FACE_SMOOTH_ALPHA
		)
		face_center_x = face_smooth_x_norm + face_smooth_width_norm / 2
		face_center_y = face_smooth_y_norm + face_smooth_height_norm / 2
		face_offset_y = face_smooth_height_norm * 1.4
		cursor_min_x_norm = face_center_x - face_smooth_width_norm * FACE_RATIO_X
		cursor_max_x_norm = face_center_x + face_smooth_width_norm * FACE_RATIO_X
		cursor_min_y_norm = face_center_y - face_smooth_height_norm * FACE_RATIO_Y + face_offset_y
		cursor_max_y_norm = face_center_y + face_smooth_height_norm * FACE_RATIO_Y + face_offset_y
		cursor_min_x_norm = max(0, cursor_min_x_norm)
		cursor_max_x_norm = min(1, cursor_max_x_norm)
		cursor_min_y_norm = max(0, cursor_min_y_norm)
		cursor_max_y_norm = min(1, cursor_max_y_norm)
	else:
		face_lost_count += 1
		if face_lost_count > FACE_LOST_LIMIT:
			#固定された範囲
			cursor_min_x_norm = margin_x
			cursor_max_x_norm = 1 - margin_x
			cursor_min_y_norm = margin_y
			cursor_max_y_norm = 1 - margin_y

	#検知する範囲の描画
	left = int(frame_width * cursor_min_x_norm)
	right = int(frame_width * cursor_max_x_norm)
	top = int(frame_height * cursor_min_y_norm)
	bottom = int(frame_height * cursor_max_y_norm)
	overlay = frame.copy()
	cv2.rectangle(overlay, (0,0), (frame_width, frame_height), (0,0,0), -1)
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
			for start, end in HAND_CONNECTIONS:
				mp_p1 = hand_landmarks[start]
				mp_p2 = hand_landmarks[end]
				frame_x1_px = int(mp_p1.x * frame_width)
				frame_y1_px = int(mp_p1.y * frame_height)
				frame_x2 = int(mp_p2.x * frame_width)
				frame_y2 = int(mp_p2.y * frame_height)
				cv2.line(frame, (frame_x1_px ,frame_y1_px), (frame_x2, frame_y2), line_color, 2)
			for landmark in hand_landmarks:
				frame_x = int(landmark.x * frame_width)
				frame_y = int(landmark.y * frame_height)
				cv2.circle(frame, (frame_x, frame_y), 5, (0, 255, 0), -1)

			#手のひらの重心に黄色の点
			indices = [0, 1, 5, 9, 13, 17]
			palm_x_norm = mean([hand_landmarks[i].x for i in indices])
			palm_y_norm = mean([hand_landmarks[i].y for i in indices])
			palm_x_px = int(palm_x_norm * frame_width)
			frame_palm_y = int(palm_y_norm * frame_height)
			cv2.circle(frame, (palm_x_px, frame_palm_y), PALM_CIRCLE_RADIUS, (0, 255, 255), -1)

			#ジェスチャーとクリック動作
			index_finger_angle = angle_3d(
				hand_landmarks[5],
				hand_landmarks[6],
				hand_landmarks[8]
			)
			middle_finger_angle = angle_3d(
				hand_landmarks[9],
				hand_landmarks[10],
				hand_landmarks[12]
			)
			ring_finger_angle = angle_3d(
				hand_landmarks[13],
				hand_landmarks[14],
				hand_landmarks[16]
			)
			pinky_finger_angle = angle_3d(
				hand_landmarks[17],
				hand_landmarks[18],
				hand_landmarks[20]
			)
			gesture = gesture_judge(
				index_finger_angle, 
				middle_finger_angle, 
				ring_finger_angle, 
				pinky_finger_angle
			)
			if gesture == 'Closed Palm':
				current_time = time.monotonic()
				if current_time - last_click_time >= CLICK_COOLDOWN:
					pyautogui.click()
					last_click_time = current_time
			if gesture == 'Victory':
				current_time = time.monotonic()
				if current_time - last_click_time >= CLICK_COOLDOWN:
					pyautogui.rightClick()
					last_click_time = current_time
			#マウス操作
			if gesture == 'Opened Palm' or gesture == 'Others':
				palm_x_norm = max(cursor_min_x_norm, min(cursor_max_x_norm, palm_x_norm))
				palm_y_norm = max(cursor_min_y_norm, min(cursor_max_y_norm, palm_y_norm))
				target_x_px = int((1 - (palm_x_norm - cursor_min_x_norm) / (cursor_max_x_norm - cursor_min_x_norm)) * screen_width)
				target_y_px = int((palm_y_norm - cursor_min_y_norm) / (cursor_max_y_norm - cursor_min_y_norm) * screen_height)
				#初回のみ
				if cursor_smooth_x_px == None:
					cursor_smooth_x_px = target_x_px
					cursor_smooth_y_px = target_y_px
					cursor_x_px = target_x_px
					cursor_y_px = target_y_px
				#EMA関数を使用して滑らかに
				cursor_smooth_x_px, cursor_smooth_y_px = ema(cursor_smooth_x_px, cursor_smooth_y_px, target_x_px, target_y_px, CURSOR_ALPHA)
				#デッドゾーン
				if in_deadzone(cursor_x_px, cursor_y_px, cursor_smooth_x_px, cursor_smooth_y_px, CURSOR_THRESHOLD) == False:
					cursor_x_px = cursor_smooth_x_px
					cursor_y_px = cursor_smooth_y_px
					pyautogui.moveTo(cursor_x_px, cursor_y_px)
	
	#確認用ウインドウの表示
	frame = cv2.flip(frame, 1)
	display = cv2.resize(frame, (int(screen_width / 2), int(screen_height /2)))
	cv2.imshow('Hand Tracking', display)
	#Qキーを押すとループから抜ける
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
#確認用ウインドウの削除
cap.release()
cv2.destroyAllWindows()