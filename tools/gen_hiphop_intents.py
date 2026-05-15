from email.mime import image
import json
import sys
import os
import cv2
from os import path as osp
import argparse
import warnings
import torch
from torchvision.transforms import v2
import numpy as np
from PIL import Image
from detectron2.config import instantiate, LazyConfig
import torchinfo
from torchinfo import summary
import time
import matplotlib.pyplot as plt
import csv

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *

PATH = "C:\\Users\janna\Downloads\ViTGaze\hiphop_v1"
WRIT_PATH = "C:\\Users\janna\Downloads\ViTGaze\hiphop_gaze\\annotations\\"
WRIT_FILE = osp.join(WRIT_PATH, "hiphop_intents.csv")
#ANNOTATION_PATH = "P1_V1.ndjson"
ROOT = "infer_out_05_12"
SKIP_FRAMES = 4
START_PTCPT = 1 

warnings.simplefilter(action="ignore", category=FutureWarning)
def head_mask_gen(head_bbox, imsize):
    head_mask = torch.zeros(imsize, dtype=torch.float32)  # Create a blank grayscale image with the specified size
    x1, y1, x2, y2 = head_bbox
    head_mask[y1:y2, x1:x2] = 1  # Set the pixel values within the head bounding box to 1 (white)
    return head_mask

def _get_intent_value(frame_annotation):
    for classification in frame_annotation.get("classifications", []):
        if classification.get("value") != "intention":
            continue

        answer = classification.get("answer")
        if isinstance(answer, dict):
            return answer.get("value") or answer.get("title") or "None"

    return "None"


def _is_gazed_object(obj_annotation):
    for classification in obj_annotation.get("classifications", []):
        answer = classification.get("answer")
        if isinstance(answer, dict):
            answer_value = str(answer.get("value", "")).lower()
            answer_title = str(answer.get("title", "")).lower()
            if answer_value in {"yes", "true", "is_gazed"} or answer_title in {
                "yes",
                "true",
                "isgazed",
            }:
                return True
            continue

        classification_value = str(classification.get("value", "")).lower()
        classification_title = str(classification.get("title", "")).lower()
        if classification_value in {"yes", "true", "is_gazed"} or classification_title in {
            "yes",
            "true",
            "isgazed",
        }:
            return True

    return False


def load_annotation(annotation_filepath: str):
    """Loads ndjson file exported from LabelBox for video annotations.
    The file contains the bounding boxes of objects detected in the video,
    along with a label of focus (isGazed) on an object with respect to the
    "head" object. This is project specific.

    Args:
        annotation_filepath (str): Relative filepath of ndjson file.
    """
    with open(annotation_filepath) as file:
        vid_annotations = [json.loads(line) for line in file.readlines()]
    
    #print(len(vid_annotations))  # Check the shape of the loaded annotations
    
    frame_annotations = []
    video_intent = "None"

    for i in range(len(vid_annotations)):  # Print the first annotation to check the structure
        intent = _get_intent_value(vid_annotations[i])
        if intent != "None":
            video_intent = intent
        gazed = "None"
        '''if i == 0:
            print(vid_annotations[i])  # Print the first annotation to check the structure'''
        
        for j in range(len(vid_annotations[i]['objects'])):
            obj_annotation = vid_annotations[i]['objects'][j]
            if _is_gazed_object(obj_annotation):
                gazed = obj_annotation.get("title") or obj_annotation.get("value") or "None"
                break

        frame_annotations.append([f"frame_{i:04d}.jpg", gazed])

    return [frame_annotations, video_intent]


def writ_annot(data, filename):
    with open(filename, mode='w', newline='') as file:
        json.dump(data, file)


def write_csv(data, filename):
    if not osp.exists(osp.dirname(filename)):
        os.makedirs(osp.dirname(filename))

    with open(filename, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["participant", "video", "annotation_file", "frames", "intent"])
        writer.writerows(data)

def writ_hiphop(filename):
    all_annotations = []

    for ptcpt in range(START_PTCPT-1, 20):
        print(f"Processing participant {ptcpt+1}...")
        for v in range(50):
            video_path = osp.join(filename, f"P{ptcpt+1}", f"V{v+1}", f"P{ptcpt+1}_V{v+1}.ndjson")
            if not osp.exists(video_path):
                print(f"Skipping missing annotation: {video_path}")
                continue

            print(f"annotation found for P{ptcpt+1}_V{v+1}, loading...")
            annotations = load_annotation(video_path)
            frame_annotations, intent = annotations
            all_annotations.append(
                [
                    f"P{ptcpt+1}",
                    f"V{v+1}",
                    f"P{ptcpt+1}_V{v+1}.ndjson",
                    json.dumps(frame_annotations),
                    intent,
                ]
            )

    write_csv(all_annotations, WRIT_FILE)
    print(f"Wrote {len(all_annotations)} video annotations to {WRIT_FILE}")

def main():
    '''annotation_filepath = osp.join(PATH, "P1_V1.ndjson")
    annotations = load_annotation(annotation_filepath)
    writ_annot(annotations, osp.join(PATH, "P1_V1.csv"))'''

    writ_hiphop(PATH)

if __name__ == "__main__":
    main()
