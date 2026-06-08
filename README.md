# hand_gesture-.py


import cv2
import math
from collections import deque
import numpy as np
import mediapipe as mp

# Utility helpers for ASL letter detection.
def distance(point1, point2):
    return math.hypot(point1.x - point2.x, point1.y - point2.y)


def hand_scale(landmarks):
    return max(
        distance(landmarks[0], landmarks[5]),
        distance(landmarks[0], landmarks[9]),
        distance(landmarks[0], landmarks[13]),
        distance(landmarks[0], landmarks[17]),
        0.1,
    )


def finger_extended(landmarks, tip, pip, mcp, scale):
    tip_pip_dist = distance(landmarks[tip], landmarks[pip])
    return (
        tip_pip_dist > scale * 0.10
        and (
            landmarks[tip].y < landmarks[pip].y < landmarks[mcp].y
            or landmarks[tip].y < landmarks[mcp].y - scale * 0.05
        )
    )


def finger_folded(landmarks, tip, pip, scale):
    return (
        landmarks[tip].y > landmarks[pip].y
        and distance(landmarks[tip], landmarks[pip]) < scale * 0.14
    )


def thumb_extended(landmarks, scale):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    is_out = distance(thumb_tip, thumb_ip) > scale * 0.12
    is_side = abs(thumb_tip.x - thumb_mcp.x) > scale * 0.18
    return is_out or is_side


def thumb_between_index_middle(index_tip, middle_tip, thumb_tip, scale):
    min_x = min(index_tip.x, middle_tip.x) - scale * 0.06
    max_x = max(index_tip.x, middle_tip.x) + scale * 0.06
    return (
        min_x <= thumb_tip.x <= max_x
        and thumb_tip.y < max(index_tip.y, middle_tip.y) + scale * 0.05
    )


def touching(landmarks, a, b, scale, threshold_factor=0.18):
    return distance(landmarks[a], landmarks[b]) < scale * threshold_factor


def fingers_together(landmarks, scale):
    return (
        distance(landmarks[8], landmarks[12]) < scale * 0.1
        and distance(landmarks[12], landmarks[16]) < scale * 0.1
        and distance(landmarks[16], landmarks[20]) < scale * 0.1
    )


def detect_asl_letter(landmarks):
    scale = hand_scale(landmarks)

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]

    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]

    index_ext = finger_extended(landmarks, 8, 6, 5, scale)
    middle_ext = finger_extended(landmarks, 12, 10, 9, scale)
    ring_ext = finger_extended(landmarks, 16, 14, 13, scale)
    pinky_ext = finger_extended(landmarks, 20, 18, 17, scale)
    thumb_ext = thumb_extended(landmarks, scale)

    index_fold = finger_folded(landmarks, 8, 6, scale)
    middle_fold = finger_folded(landmarks, 12, 10, scale)
    ring_fold = finger_folded(landmarks, 16, 14, scale)
    pinky_fold = finger_folded(landmarks, 20, 18, scale)

    all_folded = not (index_ext or middle_ext or ring_ext or pinky_ext)
    all_extended = index_ext and middle_ext and ring_ext and pinky_ext
    extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    thumb_on_side = abs(thumb_tip.x - index_mcp.x) > scale * 0.35 and thumb_tip.y > index_pip.y
    thumb_in_front = thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y
    thumb_over_fingers = thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y
    thumb_near_index = touching(landmarks, 4, 8, scale)
    thumb_near_middle = touching(landmarks, 4, 12, scale)
    thumb_near_ring = touching(landmarks, 4, 16, scale)
    thumb_near_pinky = touching(landmarks, 4, 20, scale)
    finger_grouped = fingers_together(landmarks, scale)
    thumb_folded = not thumb_ext and distance(thumb_tip, thumb_ip) < scale * 0.13

    # If thumb is touching pinky, treat as 'E' (user request)
    if thumb_near_pinky and index_ext and middle_ext and ring_ext and not pinky_ext:
        return 'E'

    if all_folded and not thumb_on_side:
        return 'A'
    if not index_ext and not middle_ext and ring_ext and pinky_ext and thumb_near_middle:
        return 'B'
    if index_ext and middle_ext and ring_ext and pinky_ext and thumb_near_index:
        return 'F'
    if all_extended and thumb_ext:
        return 'C'
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        if thumb_near_middle or thumb_near_ring:
            return 'D'
        if thumb_ext and thumb_tip.y > index_tip.y:
            return 'G'
        if thumb_ext:
            return 'L'
        return 'D'
    if all_folded and thumb_in_front:
        return 'S'
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and thumb_ext:
        if pinky_ext:
            return 'O'
        return 'G'
    if index_ext and middle_ext and ring_ext and not pinky_ext and thumb_ext:
        if thumb_near_pinky:
            return 'E' 
        return 'H'
    if pinky_ext and not index_ext and not middle_ext and not ring_ext:
        return 'I'
    if index_ext and middle_ext and not ring_ext and not pinky_ext and thumb_ext:
        if thumb_between_index_middle(index_tip, middle_tip, thumb_tip, scale):
            return 'L'
        if thumb_on_side:
            return 'K'
        if pinky_ext:
            return 'P'
        return 'L'
    if all_folded and not thumb_ext and not thumb_in_front and thumb_near_index:
        return 'T'
    if index_ext and middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        if touching(landmarks, 8, 12, scale) and extended_count == 2:
            return 'U'
        if abs(index_tip.x - middle_tip.x) > scale * 0.25:
            return 'V'
        return 'U'
    if index_ext and middle_ext and ring_ext and not pinky_ext and not thumb_ext:
        return 'W'
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        if index_fold and not middle_ext:
            return 'X'
        if thumb_ext and thumb_on_side:
            return 'Y'
        return 'D'
    if thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
        return 'Y'
    if index_ext and not middle_ext and not ring_ext and pinky_ext and thumb_ext:
        return 'O'
    if index_ext and middle_ext and not ring_ext and pinky_ext and thumb_ext and thumb_tip.y > index_tip.y:
        return 'P'
    if all_folded and thumb_ext and thumb_tip.y > wrist.y:
        return 'A'
    if all_extended and thumb_ext:
        return 'C'
    return ''



def hand_scale(landmarks):
    return max(
        distance(landmarks[0], landmarks[5]),
        distance(landmarks[0], landmarks[9]),
        distance(landmarks[0], landmarks[13]),
        distance(landmarks[0], landmarks[17]),
        0.1,
    )

def finger_extended(landmarks, tip, pip, mcp, scale):
    tip_pip_dist = distance(landmarks[tip], landmarks[pip])
    return (
        tip_pip_dist > scale * 0.10
        and (
            landmarks[tip].y < landmarks[pip].y < landmarks[mcp].y
            or landmarks[tip].y < landmarks[mcp].y - scale * 0.05
        )
    )

def finger_folded(landmarks, tip, pip, scale):
    return (
        landmarks[tip].y > landmarks[pip].y
        and distance(landmarks[tip], landmarks[pip]) < scale * 0.14
    )

def thumb_extended(landmarks, scale):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    is_out = distance(thumb_tip, thumb_ip) > scale * 0.12
    is_side = abs(thumb_tip.x - thumb_mcp.x) > scale * 0.18
    return is_out or is_side

def thumb_between_index_middle(index_tip, middle_tip, thumb_tip, scale):
    min_x = min(index_tip.x, middle_tip.x) - scale * 0.06
    max_x = max(index_tip.x, middle_tip.x) + scale * 0.06
    return (
        min_x <= thumb_tip.x <= max_x
        and thumb_tip.y < max(index_tip.y, middle_tip.y) + scale * 0.05
    )

def touching(landmarks, a, b, scale, threshold_factor=0.18):
    return distance(landmarks[a], landmarks[b]) < scale * threshold_factor

def fingers_together(landmarks, scale):
    return (
        distance(landmarks[8], landmarks[12]) < scale * 0.1
        and distance(landmarks[12], landmarks[16]) < scale * 0.1
        and distance(landmarks[16], landmarks[20]) < scale * 0.1
    )

def detect_asl_letter(landmarks):
    scale = hand_scale(landmarks)

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]

    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]

    index_ext = finger_extended(landmarks, 8, 6, 5, scale)
    middle_ext = finger_extended(landmarks, 12, 10, 9, scale)
    ring_ext = finger_extended(landmarks, 16, 14, 13, scale)
    pinky_ext = finger_extended(landmarks, 20, 18, 17, scale)
    thumb_ext = thumb_extended(landmarks, scale)

    index_fold = finger_folded(landmarks, 8, 6, scale)
    middle_fold = finger_folded(landmarks, 12, 10, scale)
    ring_fold = finger_folded(landmarks, 16, 14, scale)
    pinky_fold = finger_folded(landmarks, 20, 18, scale)

    all_folded = not (index_ext or middle_ext or ring_ext or pinky_ext)
    all_extended = index_ext and middle_ext and ring_ext and pinky_ext
    extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    thumb_on_side = abs(thumb_tip.x - index_mcp.x) > scale * 0.35 and thumb_tip.y > index_pip.y
    thumb_in_front = thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y
    thumb_over_fingers = thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y
    thumb_near_index = touching(landmarks, 4, 8, scale)
    thumb_near_middle = touching(landmarks, 4, 12, scale)
    thumb_near_ring = touching(landmarks, 4, 16, scale)
    thumb_near_pinky = touching(landmarks, 4, 20, scale)
    finger_grouped = fingers_together(landmarks, scale)
    thumb_folded = not thumb_ext and distance(thumb_tip, thumb_ip) < scale * 0.13

    # If thumb is touching pinky, treat as 'E' (user request)
    if thumb_near_pinky and index_ext and middle_ext and ring_ext and not pinky_ext:
        return 'E'

    if all_folded and not thumb_on_side:
        return 'A'
    if not index_ext and not middle_ext and ring_ext and pinky_ext and thumb_near_middle:
        return 'B'
    if index_ext and middle_ext and ring_ext and pinky_ext and thumb_near_index:
        return 'F'
    if all_extended and thumb_ext:
        return 'C'
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        if thumb_near_middle or thumb_near_ring:
            return 'D'
        if thumb_ext and thumb_tip.y > index_tip.y:
            return 'G'
        if thumb_ext:
            return 'L'
        return 'D'
    if all_folded and thumb_in_front:
        return 'S'
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and thumb_ext:
        if pinky_ext:
            return 'O'
        return 'G'
    if index_ext and middle_ext and ring_ext and not pinky_ext and thumb_ext:
        if thumb_near_pinky:
            return 'E' 
        return 'H'
    if pinky_ext and not index_ext and not middle_ext and not ring_ext:
        return 'I'
    if index_ext and middle_ext and not ring_ext and not pinky_ext and thumb_ext:
        if thumb_between_index_middle(index_tip, middle_tip, thumb_tip, scale):
            return 'L'
        if thumb_on_side:
            return 'K'
        if pinky_ext:
            return 'P'
        return 'L'
    if all_folded and not thumb_ext and not thumb_in_front and thumb_near_index:
        return 'T'
    if index_ext and middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        if touching(landmarks, 8, 12, scale) and extended_count == 2:
            return 'U'
        if abs(index_tip.x - middle_tip.x) > scale * 0.25:
            return 'V'
        return 'U'
    if index_ext and middle_ext and ring_ext and not pinky_ext and not thumb_ext:
        return 'W'
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        if index_fold and not middle_ext:
            return 'X'
        if thumb_ext and thumb_on_side:
            return 'Y'
        return 'D'
    if thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
        return 'Y'
    if index_ext and not middle_ext and not ring_ext and pinky_ext and thumb_ext:
        return 'O'
    if index_ext and middle_ext and not ring_ext and pinky_ext and thumb_ext and thumb_tip.y > index_tip.y:
        return 'P'
    if all_folded and thumb_ext and thumb_tip.y > wrist.y:
        return 'A'
    if all_extended and thumb_ext:
        return 'C'
    return ''

SUPPORTED_RIGHT_LETTERS = 'A B C D E G H I K L O P S'
SUPPORTED_LEFT_LETTERS = 'F J M N Q R T U V W X Y Z'
SUPPORTED_RIGHT_SET = set(SUPPORTED_RIGHT_LETTERS.split())
SUPPORTED_LEFT_SET = set(SUPPORTED_LEFT_LETTERS.split())
RIGHT_SHAPE_ORDER = SUPPORTED_RIGHT_LETTERS.split()
LEFT_MAPPED_LETTERS = SUPPORTED_LEFT_LETTERS.split()
RIGHT_TO_LEFT_MAP = dict(zip(RIGHT_SHAPE_ORDER, LEFT_MAPPED_LETTERS))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)

# Open WebCam (use CAP_DSHOW on Windows for better camera support)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open webcam. Try closing other camera apps or using a different camera index.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

cv2.namedWindow('MediaPipe Hand Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('MediaPipe Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

label_history = deque(maxlen=10)
stable_label = ''

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame. Check the camera connection and index.")
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_label = ''
    hand_side = ''
    if results.multi_hand_landmarks and results.multi_handedness:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness_label = results.multi_handedness[0].classification[0].label  # 'Right' or 'Left'
        hand_side = f'{handedness_label} hand'
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        )
        gesture_label = detect_asl_letter(hand_landmarks.landmark)
        if handedness_label == 'Right':
            if gesture_label not in SUPPORTED_RIGHT_SET:
                gesture_label = ''
        else:
            gesture_label = RIGHT_TO_LEFT_MAP.get(gesture_label, '')

    label_history.append(gesture_label)
    if len(label_history) == label_history.maxlen:
        most_common = max(set(label_history), key=label_history.count)
        if most_common and label_history.count(most_common) >= 6:
            stable_label = most_common

    output_label = stable_label if stable_label else gesture_label
    if not output_label:
        stable_label = ''

    label_text = f'Recognized ({hand_side}): {output_label}' if hand_side else f'Recognized: {output_label}'
    if not output_label:
        label_text = 'Recognized:'

    frame_h, frame_w = frame.shape[:2]
    max_feed_w = 900
    max_feed_h = 600
    scale = min(max_feed_w / frame_w, max_feed_h / frame_h, 1.0)
    if scale < 1.0:
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frame_h, frame_w = new_h, new_w

    screen_h = frame_h + 160
    screen_w = frame_w + 100
    screen = np.full((screen_h, screen_w, 3), 12, dtype=np.uint8)
    x_offset = (screen_w - frame_w) // 2
    y_offset = 20
    screen[y_offset:y_offset + frame_h, x_offset:x_offset + frame_w] = frame
    cv2.rectangle(
        screen,
        (x_offset - 2, y_offset - 2),
        (x_offset + frame_w + 2, y_offset + frame_h + 2),
        (255, 255, 255),
        2,
    )

    cv2.putText(
        screen,
        label_text,
        (30, y_offset + frame_h + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    supported_text = f'Right: {SUPPORTED_RIGHT_LETTERS}'
    cv2.putText(
        screen,
        supported_text,
        (30, y_offset + frame_h + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    supported_text_2 = f'Left shapes map to: {SUPPORTED_LEFT_LETTERS}'
    cv2.putText(
        screen,
        supported_text_2,
        (30, y_offset + frame_h + 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow('MediaPipe Hand Tracking', screen)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




