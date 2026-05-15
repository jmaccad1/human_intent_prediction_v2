import cv2
import json
from PIL import ImageColor
from tqdm import tqdm
import sys, os

ANNOTATION_FILEPATH = './P21.ndjson'  # Relative filepath of the annotation file

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
    
    ext_annotations = []

    for i in range(len(vid_annotations)):  # Print the first annotation to check the structure
        if i == 0:
            continue        
        
        
        #testing video2 for general case
        # Per frame analysis
        for j in range(len(vid_annotations[i]['projects']['clqc5xk9s0098070gg4qiafqi']['labels'][0]['annotations']['frames'])):
            
            x = vid_annotations[i]['projects']['clqc5xk9s0098070gg4qiafqi']['labels'][0]['annotations']['frames'][f"{j+1}"]['objects']
            for item in x:
                if(x[item]['name'] == 'Umbrella' and x[item]['classifications']):
                    print(f"Gazed at Umbrella in frame {j+1}")  # Print the number of objects in the current annotation
                    
        break

        for item in vid_annotations[i]['projects']['clqc5xk9s0098070gg4qiafqi']['labels'][0]['annotations']['frames']['1']['objects']:
            if(vid_annotations[i]['projects']['clqc5xk9s0098070gg4qiafqi']['labels'][0]['annotations']['frames']['1']['objects'][item]['name'] == 'Fruits'):
                print(vid_annotations[i]['projects']['clqc5xk9s0098070gg4qiafqi']['labels'][0]['annotations']['frames']['1']['objects'][item])  # Print the number of objects in the current annotation
        #print((vid_annotations[i]['data_row']['row_data']))  # Print the keys of the current annotation to check the structure
        break

        #for j in range(len(vid_annotations[i])):
            #print("J: ", j)
            #print(f"Frame {i}, Object {j+1}: {vid_annotations[i][j+1]}")  # Print the first annotation to check the structure
        
        #ext_annotations.append(obj_annot)

    '''head_mask = []
    head_mask = head_mask_gen(ext_annotations[0][15][:-1], (1080, 1920))  # Generate head mask using the head bounding box from the first annotation
    '''
    #head_truth = vid_annotations[0]['objects'][-1]['bbox']
    #print(head_truth)  # Print the head bounding box to check the structure

    return vid_annotations


def main():
    filepath = ANNOTATION_FILEPATH
    file = load_annotation(filepath)

    '''for i in range(len(file)):
        if i == 0:
            continue  # Skip the first annotation if it's not relevant
        print(f"Frame {i}: {file[i]}")
        break  # Print the annotations for each frame
    '''
if __name__ == "__main__":
    main()