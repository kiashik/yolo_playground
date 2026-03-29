# # from ultralytics.models.sam import SAM3SemanticPredictor

# # # Initialize predictor with configuration
# # overrides = dict(
# #     conf=0.25,
# #     task="segment",
# #     mode="predict",
# #     model="sam3.pt",
# #     half=True,  # Use FP16 for faster inference
# #     save=True,
# # )
# # predictor = SAM3SemanticPredictor(overrides=overrides)

# # # Set image once for multiple queries
# # predictor.set_image("my_images.jpg")

# # # Query with multiple text prompts
# # results = predictor(text=["tennis ball", "person"])

# # # # Works with descriptive phrases
# # # results = predictor(text=["person with red cloth", "person with blue cloth"])

# # # # Query with a single concept
# # # results = predictor(text=["a person"])

# ###############################################
# import os

# from ultralytics import SAM
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# from pathlib import Path


# # Load a model
# model = SAM("sam2.1_b.pt")

# # Display model information (optional)
# model.info()

# # Run inference
# # Run inference (disable GUI display on headless systems)
# results = model("bus.jpg", show=False)

# # # Save an annotated image with segmentations
# # os.makedirs("runs/segment", exist_ok=True)
# # results[0].save(filename="runs/segment/bus_sam_segmented.jpg")

# # Display results and wait for key press or mouse click
# for result in results:
#     # Get the annotated image (in BGR format)
#     annotated_frame = result.plot()
    
#     # Convert BGR to RGB for matplotlib
#     annotated_rgb = annotated_frame[:, :, ::-1]
    
#     # Display the image
#     plt.figure(figsize=(12, 9))
#     plt.imshow(annotated_rgb)
#     plt.axis('off')
#     plt.title('Detection Results - Press any key or click to continue')
#     plt.tight_layout()
#     plt.waitforbuttonpress()  # Wait for key press or mouse click
#     plt.close()

#################
import os

from ultralytics.models.fastsam import FastSAMPredictor

# Create FastSAMPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-s.pt", save=False, imgsz=1024)
predictor = FastSAMPredictor(overrides=overrides)

# Segment everything
everything_results = predictor("img4.jpg")

# Save annotated segmentation image
os.makedirs("runs/segment", exist_ok=True)
everything_results[0].save(filename="runs/segment/fastsam_img4_segmented.jpg")

# Optional prompt inference (bbox/point do not require CLIP)
# bbox_results = predictor.prompt(everything_results, bboxes=[[200, 200, 300, 300]])
# point_results = predictor.prompt(everything_results, points=[200, 200])

# Optional text prompt (requires openai-clip; skip if unavailable)
try:
	import clip  # type: ignore

	if hasattr(clip, "load"):
		text_results = predictor.prompt(everything_results, texts="tennis ball")
		text_results[0].save(filename="runs/segment/fastsam_img4_tennis_ball.jpg")
	else:
		print("Skipping text prompt: clip.load() not available.")
except Exception as exc:
	print(f"Skipping text prompt due to CLIP error: {exc}")