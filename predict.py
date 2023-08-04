from ultralytics import YOLO

model = YOLO("./model/yolov8m_cus.pt")  # build a new model from scratch


def predict(image):
    result = model.predict(source=image, conf=0.8)  # train the model

    return result
