
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/v8/yolov8PDNet.yaml")
    results = model.train(data='', epochs=200)
    results = model.train(resume=True)



