from inference import get_model
import supervision as sv
import sys
import cv2
from inference_sdk import InferenceHTTPClient

MODEL_ID = "faces-hlp7i/1"
API_KEY = "uTsFjzXdAOk8fIy73hmr"


image_file = "C:\\Users\\janna\\Downloads\\ViTGaze\\test_img.jpg"
image = cv2.imread(image_file)


CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

result = CLIENT.infer(image_file, model_id=MODEL_ID)


all_items = result.items()
print(all_items)  # Print all key-value pairs in the result dictionary

detections = sv.Detections.from_inference(result)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)