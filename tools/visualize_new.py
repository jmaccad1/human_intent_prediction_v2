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

PATH = "C:\\Users\janna\Downloads\ViTGaze\dataset_V1\\"
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
        
        for j in range(len(vid_annotations[i]['objects'])):
            if vid_annotations[i]['objects'][j]['classifications']:
                gaze = 1
            else:
                gaze = 0
            
            x1 = int(vid_annotations[i]['objects'][j]['bbox']['left'])
            y1 = int(vid_annotations[i]['objects'][j]['bbox']['top'])
            h = int(vid_annotations[i]['objects'][j]['bbox']['height'])
            w = int(vid_annotations[i]['objects'][j]['bbox']['width'])
            x2 = x1 + w
            y2 = y1 + h
            obj_annot.append((x1, y1, x2, y2, gaze))
        ext_annotations.append(obj_annot)

    '''head_mask = []
    head_mask = head_mask_gen(ext_annotations[0][15][:-1], (1080, 1920))  # Generate head mask using the head bounding box from the first annotation
    '''
    #head_truth = vid_annotations[0]['objects'][-1]['bbox']
    #print(head_truth)  # Print the head bounding box to check the structure

    return ext_annotations

def display_image(img1, img2, img3, img4):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(img1)
    axs[0, 0].set_title("Image 1")
    axs[0,0].axis('off')
    axs[0, 1].imshow(img2)
    axs[0, 1].set_title("Image 2")
    axs[0,1].axis('off')
    axs[1, 0].imshow(img3)
    axs[1, 0].set_title("Image 3")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(img4)
    axs[1, 1].set_title("Image 4")
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def img_visualize(img_tensor):
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    img_np = (img_np * 255).astype(np.uint8)  # Scale back to [0, 255]
    cv2.imshow("Image", img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@torch.no_grad()
def test_plot(model):
    
    #video_path = VIDEO_PATH
    #annotation_path = ANNOTATION_PATH
    root = ROOT
    start_ptcpt = START_PTCPT
    skip_frames = SKIP_FRAMES

    for ptcptnum in range(start_ptcpt-1, 20):

        for vidnum in range(50):
            
            video_path = f"{PATH}\P{ptcptnum+1}\V{vidnum+1}\P{ptcptnum+1}_V{vidnum+1}.mp4"
            annotation_path = f"{PATH}\P{ptcptnum+1}\V{vidnum+1}\P{ptcptnum+1}_V{vidnum+1}.ndjson"
            #print(annotation_path)
            print(f"---STARTING VIDEO {vidnum+1} OF PARTICIPANT {ptcptnum+1}---")  # Print a message before starting each video

            vidframes = video_to_frames(video_path)
            annotations = load_annotation(annotation_path)
            

            for i in range(0, len(vidframes), skip_frames):
                #print(f"frame {i+1}")  # Print the current frame number being processed
                
                for j in range(len((annotations[i]))):
                    if annotations[i][j][4] == 1:  
                        #print(f"{j} gazed")
                        gaze_inouts = 1
                        #x_truth = (annotations[i][j][0] + annotations[i][j][2]) / 2
                        x1_truth = annotations[i][j][0]
                        x2_truth = annotations[i][j][2]
                        y1_truth = annotations[i][j][1]
                        y2_truth = annotations[i][j][3]
                        #y_truth = (annotations[i][j][1] + annotations[i][j][3]) / 2
                        break
                    if j == len(annotations[i]) - 1:
                        x1_truth = 0 
                        y1_truth = 0
                        x2_truth = 0
                        y2_truth = 0
                        gaze_inouts = 0
                
                
                #print(f"frame {i+1}")

                # Check the structure of the extracted annotations
                img_tensor = torch.tensor(vidframes[i])
                img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
                transform = v2.Resize((434, 434))
                out = transform(img_tensor)
                out = out.unsqueeze(0)  # Add batch dimension

                #insert dummy head channel (all zeros) for testing
                head_channel = head_mask_gen(annotations[i][15][:-1], (1080, 1920))  # Shape: (batch_size, channels, height, width)
                head_channel = head_channel.permute(0, 1).float() # Resize head channel to match the input size of the model
                transform_head = v2.Resize((434, 434))
                head_channel = transform_head(head_channel.unsqueeze(0))  # Add batch dimension and channel dimension
                #print(head_channel.shape) # Check the shape of the head channel tensor

                data_dict = {
                    "images": out,
                    "head_channels": head_channel
                }

                val_gaze_heatmap_pred, inout_pred = model(data_dict)
                #print(f"Frame {i} - Gaze inout prediction: {inout_pred.item()}, Truth: {gaze_inouts}") # Print the value of the gaze_inouts prediction
                val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()
                #print('gumana?')   #Gumana parin
                
                pred_x, pred_y = argmax_pts(val_gaze_heatmap_pred[0])
                norm_p = [
                    pred_x / val_gaze_heatmap_pred[0].shape[-2],
                    pred_y / val_gaze_heatmap_pred[0].shape[-1],
                ]

                scaled_heatmap = np.array(
                        Image.fromarray(val_gaze_heatmap_pred[0]).resize(
                            (1920, 1080),  # Resize to the original image size),
                        resample=Image.BILINEAR,
                    )
                )
                norm_x1_truth = x1_truth / 1920
                norm_y1_truth = y1_truth / 1080
                norm_x2_truth = x2_truth / 1920
                norm_y2_truth = y2_truth / 1080

                gaze_out_test = []
                gaze_out_test.append([norm_x1_truth, norm_y1_truth])
                gaze_out_test.append([norm_x2_truth, norm_y2_truth]) 
                if gaze_inouts == 1:
                    gaze_out_test.append([norm_x1_truth, norm_y2_truth])
                    gaze_out_test.append([norm_x2_truth, norm_y1_truth])  # Predicted gaze point
                    gaze_out_test.append([(norm_x1_truth + norm_x2_truth) / 2, (norm_y1_truth + norm_y2_truth) / 2])  # Add the center point of the bounding box as an additional ground truth gaze point
                
                multi_hot = multi_hot_targets(gaze_out_test, (1920, 1080)) # Groundtruth gaze pts
                auc_score = auc(scaled_heatmap, multi_hot)
                #print("AUC: ", auc_score)

                all_distances = []
                for gt_gaze in gaze_out_test:
                    all_distances.append(L2_dist(gt_gaze, norm_p))
                    #print(f"GT gaze point: {gt_gaze}")
                    #print(f"Predicted point: {norm_p}")
                min_dist = min(all_distances)
                test_tensor = torch.tensor(gaze_out_test)
                mean_gt_gaze = torch.mean(test_tensor.float(), 0)
                avg_dist = L2_dist(mean_gt_gaze, norm_p)

                print(f"f {i+1} - {auc_score:.4f} | {min_dist:.4f} | {avg_dist:.4f} | {inout_pred.item():.4f} | {gaze_inouts}")  # Print the evaluation metrics for each frame
                

                with open(f"{root}/metrics_p{ptcptnum+1}_v{vidnum+1}.txt", "a+") as f:
                    writer = csv.writer(f)
                    writer.writerow([i+1, auc_score, min_dist, avg_dist.item(), inout_pred.item(), gaze_inouts])  # Write the metrics to a CSV file for each frame
                
                
                #print(norm_p)
                #display_image(vidframes[i], head_channel.squeeze(0).cpu().numpy(), scaled_heatmap, multi_hot)  # Display the original image, head channel, and heatmap
                #print("Test")
                
                #multi_hot = multi_hot_targets(data_dict["gazes"][0], data_dict["imsize"][0]) # Groundtruth gazze pts


                #print("Predicted gaze point (pixel coordinates) at frame ", i, ": ", (pred_x, pred_y), "with confidence ", inout_pred.item())  # Print the predicted gaze point and confidence score for each frame

                #os.makedirs(f"{root}", exist_ok=True)
                #draw(data_dict, scaled_heatmap, f"{root}/resultnew_{i}.png") # Save the inference result as JPEG
                #print(f"Saved result for frame {i}")  # Print a message after saving each frame's result
    
            print(f"Finished processing video {vidnum+1} of participant {ptcptnum+1}")  # Print a message after finishing each video
            #sys.exit(0) #Test first video of first participant before running the whole loop

    
    print("Finished processing all videos and participants")  # Print a message after finishing all videos and participants


def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    vid = []
    success, frame = cap.read()

    while success:
        vid.append(frame)
        success, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    return vid

def main(args):
    cfg = LazyConfig.load(args.config_file)
    model: torch.nn.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights)["model"])
    model.to(cfg.train.device).train(False)
    cfg.dataloader.val.batch_size = 1
    dataloader = instantiate(cfg.dataloader.val)
    model.eval()
    test_plot(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help="config file",
    )
    parser.add_argument(
        "--model_weights",
        "-w",
        type=str,
        help="model weights",
    )
    args = parser.parse_args()
    main(args)
