import cv2
from ultralytics import YOLO


model = YOLO("runs/detect/8x/weights/best.pt")  # pretrained YOLOv8n model
#frame = cv2.imread("20230618_184436-COLLAGE.jpg")
# model.conf = 0.80
# model.iou = 0.10

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()

    results = model([frame], stream=True)
    print(frame.shape)
    for result in results:
        for i, data in enumerate(result.boxes.data.tolist()):
            classs = result.boxes.cls.tolist()[i]

            confidence = data[4]

            print(confidence, result.names[classs])

            # xmin, ymin, xmax, ymax = (
            #     int(data[0]) - 45,
            #     int(data[1]) - 45,
            #     int(data[2]) + 35,
            #     int(data[3]) + 35,
            # )
            if confidence>0.40:
                xmin, ymin, xmax, ymax = (
                    int(data[0]),
                    int(data[1]),
                    int(data[2]),
                    int(data[3]),
                )
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Using cv22.putText() method
                frame = cv2.putText(
                    frame,
                    f"{result.names[classs]}",
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    frame = cv2.resize(frame, (900, 1000))
    cv2.imshow("ff", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()