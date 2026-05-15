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
    
    ext_annotations = []

    for i in range(len(vid_annotations)):  # Print the first annotation to check the structure
        obj_annot = []

        gaze_flag = 0
        '''if i == 0:
            print(vid_annotations[i])  # Print the first annotation to check the structure'''
        
        for j in range(len(vid_annotations[i]['objects'])):

            if vid_annotations[i]['objects'][j]['classifications']:
                #print(vid_annotations[i]['objects'][j]['value']) 
                gaze = 1
                x1 = int(vid_annotations[i]['objects'][j]['bbox']['left'])
                y1 = int(vid_annotations[i]['objects'][j]['bbox']['top'])
                h = int(vid_annotations[i]['objects'][j]['bbox']['height'])
                w = int(vid_annotations[i]['objects'][j]['bbox']['width'])
                x2 = x1 + w / 2 # Use the center of the gaze target as the gaze point
                y2 = y1 + h / 2
                gaze_flag = 1
            else:
                gaze = 0
                if (vid_annotations[i]['objects'][j]['value']) == "head":
                    #print("head detected")
                    head_x1 = int(vid_annotations[i]['objects'][j]['bbox']['left'])
                    head_y1 = int(vid_annotations[i]['objects'][j]['bbox']['top']) 
                    head_h = int(vid_annotations[i]['objects'][j]['bbox']['height'])
                    head_w = int(vid_annotations[i]['objects'][j]['bbox']['width']) 
                    head_x2 = head_x1 + head_w
                    head_y2 = head_y1 + head_h   
            
        if gaze_flag:
            obj_annot.append((head_x1, head_y1, head_x2, head_y2, x2, y2))
        else:   
            obj_annot.append((head_x1, head_y1, head_x2, head_y2, -1, -1))  # Use -1 to indicate no gaze target

        ext_annotations.append(obj_annot)

    return ext_annotations


def writ_annot(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        '''writer.writerow(
            [
                "i",
                "head_x1",
                "head_y1",
                "head_x2",
                "head_y2",
                "gaze_x2",
                "gaze_y2",
            ]
        )'''
        for i, obj_annot in enumerate(data):
            for annot in obj_annot:
                writer.writerow([f"frame_{i:04d}.jpg", *annot])


def write_csv(data, filename):
    writ_annot(data, filename)

def writ_hiphop(filename):
    for ptcpt in range(START_PTCPT-1, 20):
        print(f"Processing participant {ptcpt+1}...")
        for v in range(50):
            video_path = osp.join(filename, f"P{ptcpt+1}", f"V{v+1}", f"P{ptcpt+1}_V{v+1}.ndjson")
            annotations = load_annotation(video_path)
            path = osp.join(WRIT_PATH, f"P{ptcpt+1}", f"V{v+1}")
            if not osp.exists(path):
                os.makedirs(osp.join(WRIT_PATH, f"P{ptcpt+1}", f"V{v+1}"))
            writ_annot(annotations, osp.join(WRIT_PATH, f"P{ptcpt+1}", f"V{v+1}", f"P{ptcpt+1}_V{v+1}.txt"))

def main():
    '''annotation_filepath = osp.join(PATH, "P1_V1.ndjson")
    annotations = load_annotation(annotation_filepath)
    writ_annot(annotations, osp.join(PATH, "P1_V1.csv"))'''

    writ_hiphop(PATH)

if __name__ == "__main__":
    main()
