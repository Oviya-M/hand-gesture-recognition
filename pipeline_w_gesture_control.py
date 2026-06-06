import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import os

landmarker_path = r"C:\Users\zelam\Documents\UW\CSE 576 - Computer Vision\hand_landmarker.task"

MODEL_FILENAME = "best_model.pt" # Same folder

class GestureNet(nn.Module):
    def __init__(self, input_dim=63, num_classes=8, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers += [
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = GestureNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    class_names = checkpoint["class_names"]
    return model, class_names

def normalize_landmarks(features):
    norm = np.linalg.norm(features, ord=2, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return features / norm

def predict_gesture(model, hand_landmarks, device, class_names):
    # Extract wrist-relative coordinates (wrist becomes origin)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    coords_rel = coords - coords[0]
    features = coords_rel.flatten()
    features_norm = normalize_landmarks(features)
    input_tensor = torch.from_numpy(features_norm).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    return class_names[pred_idx], confidence

def draw_hand(frame, hand_landmarks, show_skeleton):
    h, w, _ = frame.shape
    # Convert normalized landmarks to pixel coordinates
    points = []
    for lm in hand_landmarks:
        points.append((int(lm.x * w), int(lm.y * h)))
    # Skeleton connections (21-point hand)
    connections = [
        (0,1),(0,5),(0,9),(0,13),(0,17),
        (1,2),(2,3),(3,4),
        (5,6),(6,7),(7,8),
        (9,10),(10,11),(11,12),
        (13,14),(14,15),(15,16),
        (17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    if show_skeleton:
        for a,b in connections:
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], (0,255,0), 2)
        for idx, pt in enumerate(points):
            color = (0,0,255) if idx in [4,8,12,16,20] else (255,0,0)
            cv2.circle(frame, pt, 5, color, cv2.FILLED)
    return points

grab = {"grabbed": False, "grabbed_box": None, "offset": (0, 0)}
boxes = [{"cx": 320, "cy": 240, "w": 200, "h": 200, "color": (0, 0, 255)}]

def get_box(px, py):
    for i, b in enumerate(boxes):
        left = b["cx"] - b["w"] // 2
        right = b["cx"] + b["w"] // 2
        top = b["cy"] - b["h"] // 2
        bottom = b["cy"] + b["h"] // 2
        if left <= px <= right and top <= py <= bottom:
            return i
    return None

skeleton_state = {"on": False, "ok_ct": 0, "stop_ct": 0}  
ok_ct = [0, 0]
stop_ct = [0, 0] 
heart_ct = [0, 0]
countdown_state = {"running": False, "start_time": None}
right_like_ct = {"like": 0, "dislike": 0}  
CONF_THRESH = 0.6
FRAMES_THRESH = 6
COUNTDOWN = 3

def main():
    print("Geature 'hand_heart' w/ both hands to quit, or press 'q'")
    print("Gesture 'ok' w/ both hands to toggle on skeleton drawing.")
    print("Gesture 'stop' w/ both hands to toggle off skeleton drawing.")
    print("Gesture 'fist' w/ left hand over box to grab")
    print("While grabbing box, gesture 'dislike' w/ right hand to delete")
    print("Gesture 'fist' w/ left hand and 'like w/ right hand to create box")

    shutdown = False
    show_skeleton = False

    # Print device being used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    model, class_names = load_model(model_path, device)
    print(f"Loaded gestures: {class_names}")

    # MediaPipe HandLandmarker
    base_options = python.BaseOptions(model_asset_path=landmarker_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=lambda result, output_image, timestamp_ms: None
    )
    latest_result = None
    def save_result(result, output_image, timestamp_ms):
        nonlocal latest_result
        latest_result = result
    options.result_callback = save_result
    detector = vision.HandLandmarker.create_from_options(options)

    # Webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        # Run detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detector.detect_async(mp_img, int(time.time() * 1000))

        if latest_result:

            left_hand_state  = {"gesture": None, "conf": 0.0, "px": 0, "py": 0}
            right_hand_state = {"gesture": None, "conf": 0.0, "px": 0, "py": 0}
            detected_hands = latest_result.hand_landmarks or []

            # Collect (wrist_x, index, hand_landmarks) for sorting
            detected = []
            for j, hand_landmarks in enumerate(detected_hands):
                points_tmp = draw_hand(frame, hand_landmarks, show_skeleton)
                wrist_x = points_tmp[0][0]  # wrist pixel x
                detected.append((wrist_x, j, hand_landmarks, points_tmp))

            # Sort by wrist_x, lowest  value is left wrist
            detected.sort(key=lambda t: t[0])

            # Process sorted hands and map 0->left, 1->right
            for sorted_idx, (wrist_x, orig_j, hand_landmarks, points) in enumerate(detected):
                h_index = 0 if sorted_idx == 0 else 1
                gesture, conf = predict_gesture(model, hand_landmarks, device, class_names)
                px, py = points[8]

                # write into explicit left/right state
                if h_index == 0:
                    left_hand_state["gesture"] = gesture
                    left_hand_state["conf"] = conf
                    left_hand_state["px"] = px
                    left_hand_state["py"] = py
                else:
                    right_hand_state["gesture"] = gesture
                    right_hand_state["conf"] = conf

                # display label
                wrist_x_px, wrist_y_px = points[0]
                cv2.putText(frame, f"{gesture} ({conf:.2f})",
                            (wrist_x_px - 20, wrist_y_px - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,0), 2)

                # update per-hand counts using h_index
                if gesture == "ok" and conf > CONF_THRESH:
                    ok_ct[h_index] += 1
                    # print("ok count: ", ok_ct)
                else:
                    ok_ct[h_index] = 0

                if gesture == "stop" and conf > CONF_THRESH:
                    stop_ct[h_index] += 1
                else:
                    stop_ct[h_index] = 0

            # Reset counts for missing hands
            if len(detected) < 2:
                if left_hand_state["gesture"] is None:
                    ok_ct[0] = 0
                    stop_ct[0] = 0
                if right_hand_state["gesture"] is None:
                    ok_ct[1] = 0
                    stop_ct[1] = 0

            #Gesture "OK" w/ both hands to turn on Skeleton
            if ok_ct[0] >= FRAMES_THRESH and ok_ct[1] >= FRAMES_THRESH:
                skeleton_state["on"] = True
                show_skeleton = True
                ok_ct[0], ok_ct[1] = 0, 0
                stop_ct[0], stop_ct[1] = 0, 0

            #Gesture "STOP" w/ both hands to turn off Skeleton
            if stop_ct[0] >= FRAMES_THRESH and stop_ct[1] >= FRAMES_THRESH:
                skeleton_state["on"] = False
                show_skeleton = False
                ok_ct[0], ok_ct[1] = 0, 0
                stop_ct[0], stop_ct[1] = 0, 0

            if left_hand_state["gesture"] == "fist" and left_hand_state["conf"] > CONF_THRESH:
                lx, ly = left_hand_state["px"], left_hand_state["py"]
                #grab box
                if not grab["grabbed"]:
                    b_index = get_box(lx, ly)
                    if b_index is not None:
                        grab["grabbed"] = True
                        grab["grabbed_box"] = b_index 
                        bx = boxes[b_index]["cx"]
                        by = boxes[b_index]["cy"]
                        grab["offset"] = (bx - lx, by - ly)

                    else:
                        # no box under pointer: check right hand creating gesture
                        if right_hand_state["gesture"] == "like" and right_hand_state["conf"] > CONF_THRESH:
                            right_like_ct["like"] += 1
                        else:
                            right_like_ct["like"] = 0
                        if right_like_ct["like"] >= FRAMES_THRESH:
                            # create new box in left fist
                            new_box = {"cx": int(lx), "cy": int(ly), "w": 200, "h": 200, "color": (0, 0, 255)}
                            boxes.append(new_box)
                            right_like_ct["like"] = 0
                    
                else:
                    #move active box with pointer + offset
                    b_index = grab["grabbed_box"]
                    if b_index is not None:
                        offx, offy = grab["offset"]
                        boxes[b_index]["cx"] = int(lx + offx)
                        boxes[b_index]["cy"] = int(ly + offy)

                    # gesture right-hand "dislike" to delete box (while grabbing w/ left)
                    if right_hand_state["gesture"] == "dislike" and right_hand_state["conf"] > CONF_THRESH:
                        right_like_ct["dislike"] += 1
                    else:
                        right_like_ct["dislike"] = 0
                    if right_like_ct["dislike"] >= FRAMES_THRESH:
                        # delete active box
                        b_index = grab["grabbed_box"]
                        if b_index is not None and 0 <= b_index < len(boxes):
                            del boxes[b_index]

                        grab["grabbed"] = False
                        grab["grabbed_box"] = None
                        grab["offset"] = (0, 0)
                        right_like_ct["dislike"] = 0
                    
            else:
                #release box
                grab["grabbed"] = False
                grab["grabbed_box"] = None
                right_like_ct["like"] = 0
                right_like_ct["dislike"] = 0

            #draw boxes 
            for i, b in enumerate(boxes):
                cx, cy, w_box, h_box = b["cx"], b["cy"], b["w"], b["h"]
                if grab["grabbed"] and grab["grabbed_box"] == i:
                    cv2.rectangle(frame, (cx - w_box // 2, cy - h_box // 2), (cx + w_box // 2, cy + h_box // 2), (255, 0, 0), cv2.FILLED)
                else:
                    cv2.rectangle(frame, (cx - w_box // 2, cy - h_box // 2), (cx + w_box // 2, cy + h_box // 2), b["color"], 2)

            for h_index, hand_state in enumerate((left_hand_state, right_hand_state)):
                if hand_state["gesture"] == "hand_heart" and hand_state["conf"] > CONF_THRESH:
                    heart_ct[h_index] += 1
                    # print("heart ct: ", heart_ct)
                else:
                    heart_ct[h_index] = 0

            # If both hands show "hand_heart"  start countdown
            if heart_ct[0] >= FRAMES_THRESH and heart_ct[1] >= FRAMES_THRESH:
                if not countdown_state["running"]:
                    countdown_state["running"] = True
                    countdown_state["start_time"] = time.time()
                    # print("cd run: true")

            # If countdown running but either hand lost the gesture, cancel it
            if countdown_state["running"] and (heart_ct[0] < FRAMES_THRESH or heart_ct[1] < FRAMES_THRESH):
                countdown_state["running"] = False
                countdown_state["start_time"] = None

            # Draw and evaluate countdown
            if countdown_state["running"]:
                elapsed = time.time() - countdown_state["start_time"]
                remaining = COUNTDOWN - elapsed
                # print("time remaining: ", remaining)
                if remaining > 0:
                    # draw large countdown in center
                    s = f"Shutdown in {int(np.ceil(remaining))}"
                    cfx, cfy = frame.shape[1] // 2, frame.shape[0] // 2
                    cv2.putText(frame, s, (cfx - 340, cfy + 40), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 5)
                else:
                    shutdown = True

        cv2.imshow("Hand Gesture Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or shutdown == True:
            break
        # elif key == ord('s'):
        #     show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
     main()