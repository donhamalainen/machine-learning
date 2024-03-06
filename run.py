from ultralytics import YOLO
import cv2
import cvzone
import math

camera = cv2.VideoCapture(0)

camera.set(3, 1280)
camera.set(4, 720)

model = YOLO("../Yolo-Weights/yolov8n.pt")

while True:
    succeed, image = camera.read()
    results = model(image, show=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2-x1, y2-y1
            cvzone.cornerRect(image, (x1,y1,w,h))

            conf = math.ceil((box.conf[0] * 100))/100
            cvzone.putTextRect(image, f'{conf}', (max(0, x1), max(40, y1-20)))

    # flipping = cv2.flip(image, 1)
    cv2.imshow('image', image)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# Vapautetaan kameraresurssit ja suljetaan ikkuna
camera.release()
cv2.destroyAllWindows()