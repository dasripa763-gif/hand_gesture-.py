from pathlib import Path
p = Path("RANIT-hand_gesture_detection.py")
text = p.read_text()
old = """    frame_h, frame_w = frame.shape[:2]
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

    overlay_top = y_offset + frame_h + 15
    overlay_left = 20
    overlay_right = screen_w - 20
    overlay_bottom = overlay_top + 110
    cv2.rectangle(
        screen,
        (overlay_left, overlay_top),
        (overlay_right, overlay_bottom),
        (50, 50, 50),
        cv2.FILLED,
    )
    cv2.rectangle(
        screen,
        (overlay_left, overlay_top),
        (overlay_right, overlay_bottom),
        (100, 100, 100),
        2,
    )

    cv2.putText(
        screen,
        label_text,
        (overlay_left + 15, overlay_top + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    supported_text = f'Right: {SUPPORTED_RIGHT_LETTERS}'
    cv2.putText(
        screen,
        supported_text,
        (overlay_left + 15, overlay_top + 70),
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
        (overlay_left + 15, overlay_top + 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
"""
new = """    frame_h, frame_w = frame.shape[:2]
    max_feed_w = 640
    max_feed_h = 480
    scale = min(max_feed_w / frame_w, max_feed_h / frame_h, 1.0)
    if scale < 1.0:
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frame_h, frame_w = new_h, new_w

    overlay_height = 110
    screen_h = frame_h + overlay_height + 80
    screen_w = frame_w + 100
    screen = np.full((screen_h, screen_w, 3), 12, dtype=np.uint8)
    x_offset = (screen_w - frame_w) // 2
    y_offset = overlay_height + 20
    screen[y_offset:y_offset + frame_h, x_offset:x_offset + frame_w] = frame
    cv2.rectangle(
        screen,
        (x_offset - 2, y_offset - 2),
        (x_offset + frame_w + 2, y_offset + frame_h + 2),
        (255, 255, 255),
        2,
    )

    overlay_top = 20
    overlay_left = 20
    overlay_right = screen_w - 20
    overlay_bottom = overlay_top + overlay_height
    cv2.rectangle(
        screen,
        (overlay_left, overlay_top),
        (overlay_right, overlay_bottom),
        (50, 50, 50),
        cv2.FILLED,
    )
    cv2.rectangle(
        screen,
        (overlay_left, overlay_top),
        (overlay_right, overlay_bottom),
        (100, 100, 100),
        2,
    )

    cv2.putText(
        screen,
        label_text,
        (overlay_left + 15, overlay_top + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    supported_text = f'Right: {SUPPORTED_RIGHT_LETTERS}'
    cv2.putText(
        screen,
        supported_text,
        (overlay_left + 15, overlay_top + 70),
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
        (overlay_left + 15, overlay_top + 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
"""
if old not in text:
    raise SystemExit('Old block not found')
p.write_text(text.replace(old,new))
print('patched')
