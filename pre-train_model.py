from ultralytics import YOLO



model = YOLO("SOURCE TO OFFICIAL YOLO MODEL WEIGHTS like yolov8l.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="SOURCE TO EXPERIMENT YOLO YAML", epochs=100)  # train the model
metrics = model.val(data=f"SOURCE TO EXPERIMENT YOLO YAML")
# print(metrics_pre.results_dict) 
print(metrics.results_dict) 
# success = model.export(format="onnx")  # export the model to ONNX format




