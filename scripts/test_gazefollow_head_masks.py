import argparse
import os
import random
import cv2
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from retinaface.pre_trained_models import get_model


random.seed(1)


def gaussian(x_min, y_min, x_max, y_max):
    x_min, x_max = sorted((x_min, x_max))
    y_min, y_max = sorted((y_min, y_max))
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_sigma2, y_sigma2 = (
        np.clip(np.square([x_max - x_min, y_max - y_min], dtype=float), 1, None) / 3
    )

    def _gaussian(_xs, _ys):
        return np.exp(
            -(np.square(_xs - x_mid) / x_sigma2 + np.square(_ys - y_mid) / y_sigma2)
        )

    return _gaussian


def plot_ori(label_path, data_dir):
    df = pd.read_csv(
        label_path,
        names=[
            # Original labels
            "path",
            "idx",
            "body_bbox_x",
            "body_bbox_y",
            "body_bbox_w",
            "body_bbox_h",
            "eye_x",
            "eye_y",
            "gaze_x",
            "gaze_y",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "inout",
            "meta0",
            "meta1",
        ],
        index_col=False,
        encoding="utf-8-sig",
    )
    grouped = df.groupby("path")

    output_dir = os.path.join(data_dir, "head_masks")

    for image_name, group_df in tqdm.tqdm(grouped, desc="Generating masks with annotations: "):
        if not os.path.exists(os.path.join(output_dir, image_name)):
            w, h = Image.open(os.path.join(data_dir, image_name)).size
            heatmap = np.zeros((h, w), dtype=np.float32)
            indices = np.meshgrid(
                np.linspace(0.0, float(w), num=w, endpoint=False),
                np.linspace(0.0, float(h), num=h, endpoint=False),
            )
            for _, row in group_df.iterrows():
                x_min, y_min, x_max, y_max = (
                    row["x_min"],
                    row["y_min"],
                    row["x_max"],
                    row["y_max"],
                )
                gauss = gaussian(x_min, y_min, x_max, y_max)
                heatmap += gauss(*indices)
            heatmap /= np.max(heatmap)
            heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
            output_filename = os.path.join(output_dir, image_name)
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            heatmap_image.save(output_filename)


def plot_gen(df, data_dir):
    df = df[df["score"] > 0.8]
    grouped = df.groupby("path")
    
    output_dir = os.path.join(data_dir, "head_masks")

    for image_name, group_df in tqdm.tqdm(grouped, desc="Generating masks with predictions: "):
        w, h = Image.open(os.path.join(data_dir, image_name)).size
        heatmap = np.zeros((h, w), dtype=np.float32)
        indices = np.meshgrid(
            np.linspace(0.0, float(w), num=w, endpoint=False),
            np.linspace(0.0, float(h), num=h, endpoint=False),
        )
        for index, row in group_df.iterrows():
            x_min, y_min, x_max, y_max = (
                row["x_min"],
                row["y_min"],
                row["x_max"],
                row["y_max"],
            )
            gauss = gaussian(x_min, y_min, x_max, y_max)
            heatmap += gauss(*indices)
        heatmap /= np.max(heatmap)
        heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
        output_filename = os.path.join(output_dir, image_name)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        heatmap_image.save(output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", help="Root directory of dataset")
    parser.add_argument(
        "--subset",
        help="Subset of dataset to process",
        choices=["train", "test"],
    )
    args = parser.parse_args()
    
    label_path = os.path.join(
        args.dataset_dir, args.subset + "_annotations_release.txt"
    )
    
    '''
    column_names = [
        "path",
        "idx",
        "body_bbox_x",
        "body_bbox_y",
        "body_bbox_w",
        "body_bbox_h",
        "eye_x",
        "eye_y",
        "gaze_x",
        "gaze_y",
        "bbox_x_min",
        "bbox_y_min",
        "bbox_x_max",
        "bbox_y_max",
    ]
    df = pd.read_csv(
        label_path,
        sep=",",
        names=column_names,
        usecols=column_names,
        index_col=False,
    )
    df = df.groupby("path")

    model = get_model("resnet50_2020-07-20", max_size=2048, device="cuda")
    model.eval()

    paths = list(df.groups.keys())
    csv = []
    
    #Start number
    ctr = 2
    current = ctr
    
    for path in tqdm.tqdm(paths, desc="Predicting head bboxes: "):
    
        
        if int(path[7:14]) < ctr :
            print(path[7:14])
            continue
        
        if current != int(path[7:14]):
            df = pd.DataFrame(
                csv, columns=["path", "test_text"]
            )
            df.to_csv(os.path.join(args.dataset_dir, f"{args.subset}_testhead2.csv"), index=False, header=False, mode='a')
            csv = []
            print("Printed: ", current)
            current = int(path[7:14])
            
        
        img = cv2.imread(os.path.join(args.dataset_dir, path))

        #annotations = model.predict_jsons(img)
        #print(annotations)

        '''
        #for annotation in annotations:
            #if len(annotation["bbox"]) == 0:
                #continue
        '''
        csv.append(
            [
                path,
                'Write'
            ]
        )
        
        

    print("End of Task")
    '''
    
    
    '''
    # Write csv
    df = pd.DataFrame(
        csv, columns=["path", "score", "x_min", "y_min", "x_max", "y_max"]
    )
    df.to_csv(os.path.join(args.dataset_dir, f"{args.subset}_testhead2.csv"), index=False, header=False, mode='a')
    '''
    #Uncomment this section once all head bboxes predictions done
    
    
    df = pd.read_csv(os.path.join(args.dataset_dir, f"{args.subset}_head1.csv"))
    plot_gen(df, args.dataset_dir)
    plot_ori(label_path, args.dataset_dir)
    