from ultralytics import YOLO

model=YOLO('yolo11m.yaml')

results=model.train(data='config.yaml', epochs=100, batch=16, imgsz=640, pretrained=False,optimizer='SGD',lr0=0.01,momentum=0.9,weight_decay=0.0005, name='yolo_from_scratch')

model.mode.save('path/to/ua.pt')
