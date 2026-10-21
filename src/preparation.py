import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import pyautogui
from screeninfo import get_monitors
from statistics import mean
import time
from pathlib import Path
#====================
#設定
#====================

BASE_DIR = Path(__file__).resolve().parent
#カメラ
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
#推論用の解像度
INFER_WIDTH = 640
INFER_HEIGHT = 360
#顔検出
FACE_DETECTION_BOOL = False
FACE_MODEL_PATH = str(BASE_DIR.parent / 'models' / 'blaze_face_short_range.tflite')
FACE_MIN_DETECTION_CONFIDENCE = 0.5

FACE_LOST_LIMIT = 10
FACE_SMOOTH_ALPHA = 0.1
FACE_RATIO_X = 1.2
FACE_RATIO_Y = 0.8
FACE_OFFSET_X = 0
FACE_OFFSET_Y = 1.4
#手検出
HAND_MODEL_PATH = str(BASE_DIR.parent / 'models' / 'hand_landmarker.task')
HAND_NUM = 1
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_HAND_PRESENCE_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5
HAND_CONNECTIONS = [
	(0, 1), (1, 2), (2, 3), (3, 4),
	(0, 5), (5, 6), (6, 7), (7, 8),
	(5, 9), (9, 10), (10, 11), (11, 12),
	(9, 13), (13, 14), (14, 15), (15, 16),
	(13, 17), (17, 18), (18, 19), (19, 20),
	(0, 17)
]

HAND_LOST_LIMIT = 5
#カーソル
CURSOR_ALPHA = 0.3
CURSOR_THRESHOLD = 1
CLICK_COOLDOWN = 0.3
REQUIRED_FRAMES = 3
REQUIRED_FRAMES_VICTORY = 8
DRAG_DELAY = 0.5
DOUBLE_CLICK_INTERVAL = 0.5
#ジェスチャー
OPEN_PALM_ANGLE_THRESHOLD = 120
#描画
LANDMARK_CIRCLE_RADIUS = 15
LANDMARK_CIRCLE_COLOR = (0, 255, 0)
PALM_CENTER_CIRCLE_RADIUS = 24
PALM_CENTER_CIRCLE_COLOR = (0, 255, 255)
HAND_LINE_THICKNESS = 6

#====================
#初期化
#====================

face_lost_count = 0
face_smooth_x_norm = None
face_smooth_y_norm = None
face_smooth_width_norm = None
face_smooth_height_norm = None

hand_lost_count = 0

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
margin_x = 0.2
margin_y = 0.2
cursor_allowed = True
cursor_min_x_norm = margin_x
cursor_max_x_norm = 1 - margin_x
cursor_min_y_norm = margin_y
cursor_max_y_norm = 1 - margin_y
cursor_x_px = None
cursor_y_px = None
cursor_smooth_x_px = None
cursor_smooth_y_px = None
last_click_time = 0
candidate_count = 0
pending_click_time = None
closed_palm_start_time = None
dragging = False
last_timestamp_ms = -1

thumb_angle = None
index_finger_angle = None
middle_finger_angle = None
ring_finger_angle = None
pinky_finger_angle = None
candidate_gesture = None
confirmed_gesture = "OPEN"

#モニターの情報取得
monitors = get_monitors()
primary = next((m for m in monitors if m.is_primary), monitors[0])
screen_width , screen_height= primary.width, primary.height
#OpenCVの設定
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("opened:", cap.isOpened())
#顔検出
face_detector = None
if FACE_DETECTION_BOOL:
	if not Path(FACE_MODEL_PATH).exists():
		raise FileNotFoundError(f"Face model file not found: {FACE_MODEL_PATH}")
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
if not Path(HAND_MODEL_PATH).exists():
	raise FileNotFoundError(f"Hand model file not found: {HAND_MODEL_PATH}")
hands_base_options = python.BaseOptions(
	model_asset_path=HAND_MODEL_PATH
)
hands_options = vision.HandLandmarkerOptions(
	base_options=hands_base_options,
	running_mode=vision.RunningMode.VIDEO,
	num_hands=HAND_NUM,
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
	if ab_len == 0 or cb_len == 0:
		return 0.0
	cos_angle = dot / (ab_len * cb_len)
	cos_angle = max(-1.0, min(1.0, cos_angle))
	return math.degrees(math.acos(cos_angle))
#gesture関数
def gesture_judge(index_finger_angle, middle_finger_angle, ring_finger_angle, pinky_finger_angle):
	if (
		index_finger_angle > OPEN_PALM_ANGLE_THRESHOLD
		and middle_finger_angle > OPEN_PALM_ANGLE_THRESHOLD
		and ring_finger_angle > OPEN_PALM_ANGLE_THRESHOLD
		and pinky_finger_angle > OPEN_PALM_ANGLE_THRESHOLD
	):
		gesture = 'OPEN'
	elif (
		index_finger_angle <= OPEN_PALM_ANGLE_THRESHOLD
		and middle_finger_angle <= OPEN_PALM_ANGLE_THRESHOLD
		and ring_finger_angle <= OPEN_PALM_ANGLE_THRESHOLD
		and pinky_finger_angle <= OPEN_PALM_ANGLE_THRESHOLD
	):
		gesture = 'CLOSE'
	elif (
		index_finger_angle > OPEN_PALM_ANGLE_THRESHOLD
		and middle_finger_angle > OPEN_PALM_ANGLE_THRESHOLD
		and ring_finger_angle <= OPEN_PALM_ANGLE_THRESHOLD
		and pinky_finger_angle <= OPEN_PALM_ANGLE_THRESHOLD
	):
		gesture = 'Victory'
	else:
		gesture = 'OTHERS'
	return gesture

class Point3D:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

def to_px(landmark, width, height):	
	return Point3D(
		x=landmark.x * width,
		y=landmark.y * height,
		z=landmark.z * width
	)

#メインループ
try:
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == False:
			print("frame read failed")
			break
		frame = cv2.flip(frame, 1)
		small_frame = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
		rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(
			image_format=mp.ImageFormat.SRGB,
			data=rgb
		)
		timestamp_ms = time.monotonic_ns() // 1_000_000
		if timestamp_ms <= last_timestamp_ms:
			timestamp_ms = last_timestamp_ms + 1
		last_timestamp_ms = timestamp_ms
		hands_result = hands_landmarker.detect_for_video(
			mp_image,
			timestamp_ms
		)
		frame_height, frame_width, _ = frame.shape
		#顔検出
		face_detected = False
		if FACE_DETECTION_BOOL:
			face_result = face_detector.detect_for_video(
				mp_image,
				timestamp_ms
			)
			face_detected = len(face_result.detections) > 0
		if face_detected:
			face_lost_count = 0
			detection = face_result.detections[0]
			bbox = detection.bounding_box
			face_x_norm = bbox.origin_x / INFER_WIDTH
			face_y_norm = bbox.origin_y / INFER_HEIGHT
			face_width_norm = bbox.width / INFER_WIDTH
			face_height_norm = bbox.height / INFER_HEIGHT
			#最初だけ
			if face_smooth_x_norm is None:
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
			face_offset_y = face_smooth_height_norm * FACE_OFFSET_Y
			cursor_min_x_norm = face_center_x - face_smooth_width_norm * FACE_RATIO_X
			cursor_max_x_norm = face_center_x + face_smooth_width_norm * FACE_RATIO_X
			cursor_min_y_norm = face_center_y - face_smooth_height_norm * FACE_RATIO_Y + face_offset_y
			cursor_max_y_norm = face_center_y + face_smooth_height_norm * FACE_RATIO_Y + face_offset_y
			cursor_min_x_norm = max(0, cursor_min_x_norm)
			cursor_max_x_norm = min(1, cursor_max_x_norm)
			cursor_min_y_norm = max(0, cursor_min_y_norm)
			cursor_max_y_norm = min(1, cursor_max_y_norm)
			if cursor_max_x_norm - cursor_min_x_norm < 0.05 or cursor_max_y_norm - cursor_min_y_norm < 0.05:
				cursor_min_x_norm, cursor_max_x_norm = margin_x, 1 - margin_x
				cursor_min_y_norm, cursor_max_y_norm = margin_y, 1 - margin_y
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
		dark = cv2.convertScaleAbs(frame, alpha=0.5)
		dark[top:bottom, left:right] = frame[top:bottom, left:right]
		frame = dark
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 4)

		#手のランドマークの描画、及びカーソル操作
		if hands_result.hand_landmarks:
			hand_lost_count = 0
			for i, hand_landmarks in enumerate(hands_result.hand_landmarks):
				handedness = hands_result.handedness[i][0]
				handedness_name = handedness.category_name
				#右手 -> 赤, 左手 -> 青
				if handedness_name == 'Left':
					line_color = (0, 0, 255)
					print("Right hand detected")
				else:
					line_color = (255, 0, 0)
					print("Left hand detected")
				#点と線の描画
				for start, end in HAND_CONNECTIONS:
					mp_p1 = hand_landmarks[start]
					mp_p2 = hand_landmarks[end]
					frame_x1_px = int(mp_p1.x * frame_width)
					frame_y1_px = int(mp_p1.y * frame_height)
					frame_x2 = int(mp_p2.x * frame_width)
					frame_y2 = int(mp_p2.y * frame_height)
					cv2.line(frame, (frame_x1_px ,frame_y1_px), (frame_x2, frame_y2), line_color, HAND_LINE_THICKNESS)
				for landmark in hand_landmarks:
					frame_x = int(landmark.x * frame_width)
					frame_y = int(landmark.y * frame_height)
					cv2.circle(frame, (frame_x, frame_y), LANDMARK_CIRCLE_RADIUS, LANDMARK_CIRCLE_COLOR, -1)

				#手のひらの重心に黄色の点
				indices = [0, 1, 5, 9, 13, 17]
				palm_x_norm = mean([hand_landmarks[i].x for i in indices])
				palm_y_norm = mean([hand_landmarks[i].y for i in indices])
				palm_x_px = int(palm_x_norm * frame_width)
				frame_palm_y = int(palm_y_norm * frame_height)
				cv2.circle(frame, (palm_x_px, frame_palm_y), PALM_CENTER_CIRCLE_RADIUS, PALM_CENTER_CIRCLE_COLOR, -1)

				#ジェスチャーとクリック動作
				index_finger_angle = angle_3d(
					to_px(hand_landmarks[5], frame_width, frame_height),
					to_px(hand_landmarks[6], frame_width, frame_height),
					to_px(hand_landmarks[8], frame_width, frame_height)
				)
				middle_finger_angle = angle_3d(
					to_px(hand_landmarks[9], frame_width, frame_height),
					to_px(hand_landmarks[10], frame_width, frame_height),
					to_px(hand_landmarks[12], frame_width, frame_height)
				)
				ring_finger_angle = angle_3d(
					to_px(hand_landmarks[13], frame_width, frame_height),
					to_px(hand_landmarks[14], frame_width, frame_height),
					to_px(hand_landmarks[16], frame_width, frame_height)
				)
				pinky_finger_angle = angle_3d(
					to_px(hand_landmarks[17], frame_width, frame_height),
					to_px(hand_landmarks[18], frame_width, frame_height),
					to_px(hand_landmarks[20], frame_width, frame_height)
				)
				current_gesture = gesture_judge(
					index_finger_angle, 
					middle_finger_angle, 
					ring_finger_angle, 
					pinky_finger_angle
				)
				if current_gesture == candidate_gesture:
					candidate_count += 1
				else:
					candidate_count = 1
					candidate_gesture = current_gesture
				if candidate_gesture == 'Victory':
					required_frames = REQUIRED_FRAMES_VICTORY
				else:
					required_frames = REQUIRED_FRAMES
				if candidate_count >= required_frames:
					if confirmed_gesture != candidate_gesture:
						confirmed_gesture = candidate_gesture
						if confirmed_gesture == 'OPEN':
							if dragging:
								pyautogui.mouseUp()
								dragging = False
							if closed_palm_start_time is not None:
								elapsed_time = time.monotonic() - closed_palm_start_time
								if elapsed_time < DRAG_DELAY:
									now = time.monotonic()
									if pending_click_time is not None and now - pending_click_time <= DOUBLE_CLICK_INTERVAL:
										pyautogui.doubleClick()
										pending_click_time = None
									else:
										pending_click_time = now
							closed_palm_start_time = None
						elif confirmed_gesture == 'CLOSE' and not dragging:
							closed_palm_start_time = time.monotonic()
						elif confirmed_gesture == 'Victory':
							pending_click_time = None
							closed_palm_start_time = None
							current_time = time.monotonic()
							if current_time - last_click_time >= CLICK_COOLDOWN:
								pyautogui.rightClick()
								last_click_time = current_time
				if confirmed_gesture == 'CLOSE' and not dragging and closed_palm_start_time is not None:
					elapsed = time.monotonic() - closed_palm_start_time
					if elapsed >= DRAG_DELAY:
						pyautogui.mouseDown()
						dragging = True

				#マウス操作
				cursor_allowed = dragging or (confirmed_gesture == 'OPEN' and current_gesture == 'OPEN' and pending_click_time is None)
				if cursor_allowed == True:
					palm_x_norm = max(cursor_min_x_norm, min(cursor_max_x_norm, palm_x_norm))
					palm_y_norm = max(cursor_min_y_norm, min(cursor_max_y_norm, palm_y_norm))
					target_x_px = int(((palm_x_norm - cursor_min_x_norm) / (cursor_max_x_norm - cursor_min_x_norm)) * screen_width)
					target_y_px = int((palm_y_norm - cursor_min_y_norm) / (cursor_max_y_norm - cursor_min_y_norm) * screen_height)
					#初回のみ
					if cursor_smooth_x_px is None:
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
		else:
			hand_lost_count += 1
			if hand_lost_count > HAND_LOST_LIMIT:
				if dragging:
					pyautogui.mouseUp()
					dragging = False
				closed_palm_start_time = None
				candidate_gesture = None
				candidate_count = 0
				confirmed_gesture = "OPEN"
				cursor_smooth_x_px = None
				cursor_smooth_y_px = None
		#保留中の単クリックを確定
		if pending_click_time is not None and time.monotonic() - pending_click_time > DOUBLE_CLICK_INTERVAL:
			pyautogui.leftClick()
			pending_click_time = None
		#確認用ウインドウの表示
		display = cv2.resize(frame, (screen_width, screen_height))
		cv2.imshow('Hand Tracking', display)
		#Qキーを押すとループから抜ける
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
finally:
	if dragging:
		pyautogui.mouseUp()
		dragging = False
	#確認用ウインドウの削除
	cap.release()
	cv2.destroyAllWindows()
	hands_landmarker.close()
	if face_detector is not None:
		face_detector.close()