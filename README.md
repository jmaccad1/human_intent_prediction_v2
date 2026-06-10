# Head-Object CNN and Gaze Predicting Vision Transformers in Improving Human Intent Prediction
This repo contains all the source code developed in the project [Head-Object CNN and Gaze Predicting Vision
Transformers in Improving Human Intent Prediction](https://drive.google.com/file/d/1DJcL3GpuNyybPnSiOzKAW_QO-8Mo9XC6/view?usp=sharing). As an overview, the system is composed of two key components: the Gazed Object Detector (Cascaded ViTGaze and yolov8n models) and the Intent Classifier.
- The ViTGaze (Gaze Detection Model) is in the `ViTGaze` folder.
- The Object Detector is in the `yolov8n` folder.
- The Intent Classifier is in the `intent_classifier` folder.
- The Overall System **for inference** is in the `multi-stage_human_intent_classifier_system` folder.

# Datasets, Models & Outputs
The datasets, models, and output .csv files that was used in this project can be accessed [here](https://drive.google.com/drive/u/1/folders/0AL0QhGcL3ipaUk9PVA). The files for creating the appropriate hiphop_gaze & yolov8 dataset folder schema are in the `tools` folder.

# Recommendations
1. Add more video samples for the model to be a better generalized intent predictor and gazed object detector and eliminate dependence on outlier data
2. Develop the weights of the ViTGaze from scratch (as opposed to finetuning from the VideoAttentionTarget checkpoint)

# Acknowledgments
This capstone project is based on the state-of-the-art gaze following model [ViTGaze](https://github.com/hustvl/ViTGaze/), object detection model [yolov8n](https://github.com/ultralytics/ultralytics) as well as parts of the implementation by the [human_intent_prediction](https://github.com/jbramos9/human_intent_prediction)
