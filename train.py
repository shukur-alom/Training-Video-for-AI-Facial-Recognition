from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)



# Use the model
model.train(data="data.yaml", epochs=int(input("Epoch Size : ")), pretrained=True, imgsz=int(input("Img Size : ")))  # train the model
