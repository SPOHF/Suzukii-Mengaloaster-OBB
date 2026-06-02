from ultralytics import YOLO
import cv2

#Replace path with your preferred run
model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="path/to/picture.jpg",
    conf=0.25,
    save=False
)

for r in results:
    img = r.plot()

    cv2.imshow("YOLO Result", img)
    cv2.waitKey(0)   # 
    cv2.destroyAllWindows()