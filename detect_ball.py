#!/home/ashik/venvs/yolo/bin/python 
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Load a model
# model = YOLO("my_models/yolo11n_last_tennis_ball_eudyi_xwxjf.pt")       # 6ms on gpu, 34 ms on cpu
# model = YOLO("my_models/yolo11n_last_tennis_ball_eudyi_xwxjf.onnx")     # 6ms on gpu, 41 ms on cpu   
# model = YOLO("my_models/yolo11n_last_tennis_ball_eudyi_xwxjf_openvino_model")     # 16ms on gpu, # 18 ms on cpu. woudl be better on intel cpu
# model = YOLO("my_models/yolo11n_last_tennis_ball_eudyi_xwxjf.engine")     # 3 ms on gpu, does not run on cpu.

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source=0,          # webscam
                        device='cpu',       # use GPU for inference
                        conf=0.25,
                        imgsz=640,
                        max_det=1,
                        stream=True,       # important for live camera
                        show=False,
                        verbose=False
                    )
# results = model.predict(source=0, conf=0.25, 
#                                     imgsz=640, half=False, device="cuda:0", 
#                                     max_det=1, visualize=False, show_boxes=True, 
#                                     stream=False, show=False)

# Display results and wait for key press or mouse click
for result in results:
    print(result)

    annotated = result.plot()
    cv2.imshow("Detection Results", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()