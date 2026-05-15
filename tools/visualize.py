import sys
import os
from os import path as osp
import argparse
import warnings
import torch
import numpy as np
from PIL import Image
from detectron2.config import instantiate, LazyConfig

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *

import torchinfo
from torchinfo import summary

warnings.simplefilter(action="ignore", category=FutureWarning)

def img_visualize(img_tensor, is_tensor=True):
    if is_tensor:
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
        img_np = (img_np * 255).astype(np.uint8)  # Scale back to [0, 255]
    else:
        img_np = img_tensor
    cv2.imshow("Image", img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_dot(img_tensor, point, color=(0, 255, 0), radius=20):
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    img_np = (img_np * 255).astype(np.uint8)
    # Print the dimensions of the image
    x = int(point[0] * img_np.shape[1])  # Scale x to image width
    y = int(point[1] * img_np.shape[0])  # Scale y to image height
    #print(f"Dot coordinates: ({x}, {y})")
    disp_img = img_np.copy()  # Create a copy of the image to draw on
    cv2.circle(disp_img, (x, y), radius, color, -1)  # Draw filled circle
    cv2.imshow("Image with Dot", disp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@torch.no_grad()
def test_plot(model, dataloader):
    for i, data in enumerate(dataloader, start=1):
        
        if i > 1:
            break  # Limit to the first 1 sample for testing
        #knowing tensor values
        
        '''
        for key, tensor in data.items():
            print(f"{key}: {tensor.shape}")

        print("Dim check complete.")
        #head_channels_np = data['head_channels'].detach().cpu().numpy()
        
        heatmaps_np = data['heatmaps'].detach().cpu().numpy()
        cv2.imshow("Heatmaps", heatmaps_np[0])  # Show the first heatmap
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #summary(model)  # Print the model summary with the correct input size
        #print(model)    
        #print(f"imsize value: {data['imsize']}") # imsize value: tensor([[500, 375]], dtype=torch.int32) '''
        
        '''#img_visualize(data['head_channels'][0])  # Visualize the head channel as an image
        print(max(data['head_channels'][0].flatten()))  # Print the maximum value in the head channel tensor
        print(min(data['head_channels'][0].flatten()))  # Print the minimum value in the head channel tensor'''

        #print(f"gazes value: {data['gazes']}") # gazes value: tensor([[[0.5000, 0.5000], [0.6000, 0.4000], [0.4000, 0.6000]]])
        #print(f"gazes_inouts value: {data['gaze_inouts']}") # gazes_inouts value: tensor([[1., 1., 1.]])
        #print(f"imsize value: {data['imsize']}") # imsize value: tensor([[500, 375]], dtype=torch.int32)

        val_gaze_heatmap_pred, bl_box = model(data)
        #print(bl_box.item()) # Print the value

        val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()
        #print("Heatmap pred: ",val_gaze_heatmap_pred.shape)

        # remove padding and recover valid ground truth points
        valid_gaze = data["gazes"][0]
        valid_gaze = valid_gaze[valid_gaze != -1].view(-1, 2)

        test_valid_gaze = valid_gaze.cpu().numpy()
        #draw_dot(data["images"][0], test_valid_gaze[0])  # Visualize the first valid gaze point on the original image

        #print(valid_gaze)
        #print(valid_gaze.shape)
        #print(valid_gaze) # Print the valid gaze points after removing padding

        #print(data['gaze_inouts'][0]) # Print the gaze_inouts tensor to check which gaze points are valid (1 for valid, 0 )
        

        # AUC: area under curve of ROC
        multi_hot = multi_hot_targets(data["gazes"][0], data["imsize"][0])
        #img_visualize(data["head_masks"][0], is_tensor=True)
        #img_visualize(data["head_channels"][0], is_tensor=True)  # Visualize the multi-hot target as an image
        #multi - hot ---- multiple pts to be predicted, not just one, so we create a multi-hot map where each valid gaze point corresponds to a hot pixel (value of 1) in the target map. This allows us to evaluate the model's performance in predicting multiple gaze points simultaneously.
        

        pred_x, pred_y = argmax_pts(val_gaze_heatmap_pred[0])
        norm_p = [
            pred_x / val_gaze_heatmap_pred[0].shape[-2],
            pred_y / val_gaze_heatmap_pred[0].shape[-1],
        ]
        scaled_heatmap = np.array(
            Image.fromarray(val_gaze_heatmap_pred[0]).resize(
                data["imsize"][0],
                resample=Image.BILINEAR,
            )
        )

        

        auc_score = auc(scaled_heatmap, multi_hot)
        # min distance: minimum among all possible pairs of <ground truth point, predicted point>
        all_distances = []
        for gt_gaze in valid_gaze:
            print(f"GT gaze point: {gt_gaze}")
            print(f"Predicted point: {norm_p}")
            all_distances.append(L2_dist(gt_gaze, norm_p))
        min_dist = min(all_distances)
        # average distance: distance between the predicted point and human average point
        mean_gt_gaze = torch.mean(valid_gaze, 0)
        avg_dist = L2_dist(mean_gt_gaze, norm_p)
        good_case = auc_score > 0.995 and min_dist < 0.005
        bad_case = auc_score < 0.5 or min_dist > 0.4

        print(f"Sample {i}: AUC={auc_score:.4f}, Min Dist={min_dist:.4f}, Avg Dist={avg_dist:.4f}")
        sys.exit(0) # Exit after processing the first sample for testing


        if good_case or bad_case:
            root = "vis_output/good_cases/" if good_case else "vis_output/bad_cases/"
            print(f"{i}: {auc_score}\t{min_dist}\t{avg_dist}")
            os.makedirs(f"{root}{i}", exist_ok=True)
            draw_origin_img(data, f"{root}{i}/origin.png")
            draw(data, scaled_heatmap, f"{root}{i}/result.png")
            # normalize heatmap to highlight the peak
            # useful for checking whether the model has learned to focus on the right region
            # not for quantitative evaluation
            scaled_heatmap = (
                scaled_heatmap - scaled_heatmap.min()
            ) / scaled_heatmap.ptp()
            draw(data, scaled_heatmap, f"{root}{i}/normed_result.png")
            #out_dict = model.forward_backbone(data)
            
            
            out_dict = model.forward(data)
            
            
            attention_maps = [
                attn_map
                for attn_map in out_dict[0][0].cpu().detach().numpy()
            ]
            for a_i, attn_map in enumerate(attention_maps):
                attn_map = np.array(
                    Image.fromarray(attn_map).resize(
                        data["imsize"][0], resample=Image.BILINEAR
                    )
                )
                attn_map = (attn_map - attn_map.min()) / attn_map.ptp()
                draw(data, attn_map, f"{root}{i}/amap{a_i}.png")


def main(args):
    cfg = LazyConfig.load(args.config_file)
    model: torch.nn.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights)["model"])
    model.to(cfg.train.device).train(False)
    cfg.dataloader.val.batch_size = 1
    dataloader = instantiate(cfg.dataloader.val)
    model.eval()
    test_plot(model, dataloader)


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
