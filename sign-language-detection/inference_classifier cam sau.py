import cv2
import pickle
import mediapipe as mp
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

model = pickle.load(open('./model.p', 'rb'))['model']
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'A', 2: 'B', 3: 'B', 4: 'C', 5: 'C', 6: 'G', 7: 'H', 8: 'I', 
               9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 
               17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

last_prediction, start_time = {0: None, 1: None}, {0: None, 1: None}
final_text = ""

def preprocess_hand_view(hand_crop):
    h, w, _ = hand_crop.shape
    scale = min(300 / w, 300 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    hand_crop_resized = cv2.resize(hand_crop, (new_w, new_h))
    hand_view = np.ones((300, 300, 3), np.uint8) * 0
    x_offset, y_offset = (300 - new_w) // 2, (300 - new_h) // 2
    hand_view[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = hand_crop_resized
    return hand_view

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    hands_data, hand_view = [], np.ones((300, 300, 3), np.uint8) * 255

    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, hand_handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_side = hand_handedness.classification[0].label 
            for connection in mp_hands.HAND_CONNECTIONS:
                x1, y1 = int(hand_landmarks.landmark[connection[0]].x * W), int(hand_landmarks.landmark[connection[0]].y * H)
                x2, y2 = int(hand_landmarks.landmark[connection[1]].x * W), int(hand_landmarks.landmark[connection[1]].y * H)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * W), int(landmark.y * H)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            data_aux = []
            x_, y_ = [lm.x for lm in hand_landmarks.landmark], [lm.y for lm in hand_landmarks.landmark]
            
            for i in range(len(hand_landmarks.landmark)):
                data_aux.extend([hand_landmarks.landmark[i].x - min(x_), hand_landmarks.landmark[i].y - min(y_)])

            data_aux.extend([0] * (84 - len(data_aux)))

            x1, y1 = int(min(x_) * W) - 20, int(min(y_) * H) - 20
            x2, y2 = int(max(x_) * W) + 20, int(max(y_) * H) + 20
            hands_data.append((x1, y1, x2, y2, data_aux, idx, hand_side))
        
        hands_data.sort(key=lambda x: x[0])

        for idx, (x1, y1, x2, y2, data_aux, hand_id, hand_side) in enumerate(hands_data):
            predicted_label = int(model.predict([np.asarray(data_aux)])[0])
            predicted_character = labels_dict[predicted_label]
            
            if predicted_character != last_prediction[hand_id]:
                last_prediction[hand_id] = predicted_character
                start_time[hand_id] = time.time()

            if start_time[hand_id] and time.time() - start_time[hand_id] > 2:
                final_text += predicted_character
                start_time[hand_id] = None
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (162, 32, 204), 2)
            cv2.putText(frame, f"{predicted_character} ({hand_side})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (162, 32, 204), 2, cv2.LINE_AA)

            if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                hand_crop = frame[y1:y2, x1:x2]
                hand_view = preprocess_hand_view(hand_crop)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, H - 60), (W - 10, H - 10), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, "Text: " + final_text, (20, H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Hand Recognition', frame)
    cv2.imshow('Hand View', hand_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
