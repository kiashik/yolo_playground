from ultralytics import YOLO
from pathlib import Path


def perform_validation(models, data_yaml, device):
    rows = []       # stores val results

    metrics_to_save = [
                "model",
                "dataset_version",
                "device",
                "weights_type",
                "precision",
                "recall",
                "mAP50",
                "mAP50-95",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
                "total_ms",
            ]
    
    for model_path in models:
        print(f"\nModel path: {model_path}, Device: {device}")

        compatible, reason = is_compatible(model_path, device)

        if not compatible:
            print(f"Model {model_path} is not compatible with device {device}: {reason}")
            rows.append([
                    Path(model_path).name,               # model
                    Path(data_yaml).parent.name,         # dataset_version
                    device,                              # device
                    "best" if "best" in Path(model_path).stem else
                    "last" if "last" in Path(model_path).stem else
                    "other",                             # weights_type
                    "N/A",                             # precision
                    "N/A",                             # recall
                    "N/A",                             # mAP50
                    "N/A",                             # mAP50-95
                    "N/A",                             # preprocess_ms
                    "N/A",                             # inference_ms
                    "N/A",                             # postprocess_ms
                    "N/A",                             # total_ms
                ])

            continue

        try:
            model = YOLO(model_path, task="detect")
            metrics = model.val(data=data_yaml, device=device)

            rows.append([
            Path(model_path).name,                         # model
            Path(data_yaml).parent.name,                   # dataset_version
            device,                                        # device
            "best" if "best" in Path(model_path).stem else
            "last" if "last" in Path(model_path).stem else
            "other",                                       # weights_type
            float(metrics.box.mp),                         # precision
            float(metrics.box.mr),                         # recall
            float(metrics.box.map50),                      # mAP50
            float(metrics.box.map),                        # mAP50-95
            float(metrics.speed["preprocess"]),            # preprocess_ms
            float(metrics.speed["inference"]),             # inference_ms
            float(metrics.speed["postprocess"]),           # postprocess_ms
            float(
                metrics.speed["preprocess"]
                + metrics.speed["inference"]
                + metrics.speed["postprocess"]
            ),                                             # total_ms
        ])
        except Exception as e:
            print(f"Validation failed for {model_path}: {e}")
            rows.append([
                    Path(model_path).name,               # model
                    Path(data_yaml).parent.name,         # dataset_version
                    device,                              # device
                    "best" if "best" in Path(model_path).stem else
                    "last" if "last" in Path(model_path).stem else
                    "other",                             # weights_type
                    "ERROR",                             # precision
                    "ERROR",                             # recall
                    "ERROR",                             # mAP50
                    "ERROR",                             # mAP50-95
                    "ERROR",                             # preprocess_ms
                    "ERROR",                             # inference_ms
                    "ERROR",                             # postprocess_ms
                    "ERROR",                             # total_ms
                ])

    if device=="cpu":
        txt_name = f"model_comparison_cpu_proper_legion.txt"
    elif device.startswith("cuda"):
        txt_name = f"model_comparison_{device.replace(':', '_')}_proper_legion.txt"
    
    with open(txt_name, "w") as f:
        f.write(
            f"{'model':<45} "
            f"{'dataset_version':<25} "
            f"{'device':<10} "
            f"{'weights_type':<12} "
            f"{'precision':>10} "
            f"{'recall':>10} "
            f"{'mAP50':>10} "
            f"{'mAP50-95':>12} "
            f"{'prep(ms)':>10} "
            f"{'infer(ms)':>10} "
            f"{'post(ms)':>10} "
            f"{'total(ms)':>10}\n"
        )
        f.write("-" * 167 + "\n")

        for row in rows:
            if row[4] == "ERROR":
                f.write(
                    f"{row[0]:<45} "
                    f"{row[1]:<25} "
                    f"{row[2]:<10} "
                    f"{row[3]:<12} "
                    f"{'ERROR':>10} "
                    f"{'ERROR':>10} "
                    f"{'ERROR':>10} "
                    f"{'ERROR':>12} "
                    f"{'ERROR':>10} "
                    f"{'ERROR':>10} "
                    f"{'ERROR':>10} "
                    f"{'ERROR':>10}\n"
                )
            elif row[4] == "N/A":
                f.write(
                    f"{row[0]:<45} "
                    f"{row[1]:<25} "
                    f"{row[2]:<10} "
                    f"{row[3]:<12} "
                    f"{'N/A':>10} "
                    f"{'N/A':>10} "
                    f"{'N/A':>10} "
                    f"{'N/A':>12} "
                    f"{'N/A':>10} "
                    f"{'N/A':>10} "
                    f"{'N/A':>10} "
                    f"{'N/A':>10}\n"
                )
            else:
                f.write(
                    f"{row[0]:<45} "
                    f"{row[1]:<25} "
                    f"{row[2]:<10} "
                    f"{row[3]:<12} "
                    f"{row[4]:>10.6f} "
                    f"{row[5]:>10.6f} "
                    f"{row[6]:>10.6f} "
                    f"{row[7]:>12.6f} "
                    f"{row[8]:>10.3f} "
                    f"{row[9]:>10.3f} "
                    f"{row[10]:>10.3f} "
                    f"{row[11]:>10.3f}\n"
                )



def is_compatible(model_path: str, device: str) -> tuple[bool, str]:
    backend = get_backend(model_path)

    if backend == "pt":
        return True, ""
    if backend == "onnx":
        return True, ""
    if backend == "engine":
        if device.startswith("cuda"):
            return True, ""
        return False, "TensorRT .engine is GPU-only in this setup"
    if backend == "openvino":
        if device == "cpu":
            return True, ""
        return False, "OpenVINO is being limited to CPU in this script"
    return False, f"Unsupported or unknown backend for path: {model_path}"



def get_backend(model_path: str) -> str:
    p = Path(model_path)
    name = p.name.lower()

    if p.is_dir() and "openvino" in name:
        return "openvino"
    if name.endswith(".pt"):
        return "pt"
    if name.endswith(".onnx"):
        return "onnx"
    if name.endswith(".engine"):
        return "engine"
    if "openvino_model" in name:
        return "openvino"
    return "unknown"


if __name__ == "__main__":
    my_models_26n = [

        # my models below
        "yolo_models/my_yolo_models/yolo26n_my_ds_v2_best.pt",
        "yolo_models/my_yolo_models/yolo26n_my_ds_v2_last.pt",
        "yolo_models/my_yolo_models/yolo26n_my_ds_v2_best.onnx",
        "yolo_models/my_yolo_models/yolo26n_my_ds_v2_last.onnx",
        "yolo_models/my_yolo_models/yolo26n_my_ds_v2_best_openvino_model",
        "yolo_models/my_yolo_models/yolo26n_my_ds_v2_last_openvino_model",
        "yolo_models/my_yolo_models/yolo26n_my_ds_v2_best.engine",
        "yolo_models/my_yolo_models/yolo26n_my_ds_v2_last.engine",
    ]

    my_models_11n = [
        "yolo_models/yolo11n_last_tennis_ball_eudyi_xwxjf.pt",
        "yolo_models/yolo11n_last_tennis_ball_eudyi_xwxjf.onnx",
        "yolo_models/yolo11n_last_tennis_ball_eudyi_xwxjf_openvino_model",
        "yolo_models/yolo11n_last_tennis_ball_eudyi_xwxjf.engine",

        # my models below
        "yolo_models/my_yolo_models/yolo11n_my_ds_v2_best.pt",
        "yolo_models/my_yolo_models/yolo11n_my_ds_v2_last.pt",
        "yolo_models/my_yolo_models/yolo11n_my_ds_v2_best.onnx",
        "yolo_models/my_yolo_models/yolo11n_my_ds_v2_last.onnx",
        "yolo_models/my_yolo_models/yolo11n_my_ds_v2_best_openvino_model",
        "yolo_models/my_yolo_models/yolo11n_my_ds_v2_last_openvino_model",
        "yolo_models/my_yolo_models/yolo11n_my_ds_v2_best.engine",
        "yolo_models/my_yolo_models/yolo11n_my_ds_v2_last.engine",
    ]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # data_yaml = "tennis-ball-validation-1_26/data.yaml"    
    # print("\n=== Validating YOLO26n on CUDA ===")
    # perform_validation(my_models_26n, data_yaml, "cuda:0")
    # print("\n=== Validating YOLO26n on CPU ===")
    # perform_validation(my_models_26n, data_yaml, "cpu")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    data_yaml = "tennis-ball-validation-1_11/data.yaml"    
    # print("\n=== Validating YOLO11n on CUDA ===")
    # perform_validation(my_models_11n, data_yaml, "cuda:0")
    print("\n=== Validating YOLO11n on CPU ===")
    perform_validation(my_models_11n, data_yaml, "cpu")
    # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++