import argparse
import os
import random
import cv2
import numpy as np
import pandas as pd
import tqdm
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="File location of test split")
    args = parser.parse_args()
    df = pd.read_csv(os.path.join(args.filename))
    print(df)
    df.to_csv(os.path.join("C:\\Users\janna\Downloads\ViTGaze\pythex\mod_text.csv"), index=False, header=False)
    
