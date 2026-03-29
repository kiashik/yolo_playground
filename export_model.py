from ultralytics import YOLO

# Load a model
my_models = [
    "yolo_models/my_yolo_models/yolo26n_my_ds_v2_best.pt",
    "yolo_models/my_yolo_models/yolo26n_my_ds_v2_last.pt",


    "yolo_models/my_yolo_models/yolo11n_my_ds_v2_best.pt",
    "yolo_models/my_yolo_models/yolo11n_my_ds_v2_last.pt",
]

# 
for model_path in my_models:
    model = YOLO(model_path)

    # Export the model
    model.export(format="onnx", imgsz=640)
    print(f"Exported {model_path} to ONNX format.")
    print("="*80)

    # model.export(format="OpenVINO", imgsz=640)
    # print(f"Exported {model_path} to OpenVINO format.")
    # print("="*80)

    # model.export(format="engine", imgsz=640)
    # print(f"Exported {model_path} to TensorRT format.")
    # print("="*80)
