import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 2
dataset_size = 200

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    folder_path = os.path.join(DATA_DIR, str(j))

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f'THU THAP DATA CHO NHÃN {j} (Nhấn "Q" để bắt đầu)')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'AN "Q" DE THU THAP DATA {j}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(folder_path, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
