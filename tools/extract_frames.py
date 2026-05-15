
# Tool to extract frames from videos in the Hiphop dataset for ViTGaze finetuning

import cv2
import json
import os 
import argparse

START_PTCPT = 4
END_PTCPT = 20

def extract_frames(video_path, output_dir):
    #assert(os.path.exists(output_dir))

    for ptcpt in range(START_PTCPT-1, END_PTCPT):
        for i in range(50):
            vid_path = f"{video_path}\P{ptcpt+1}\V{i+1}\P{ptcpt+1}_V{i+1}.mp4"
            out_dir = f"{output_dir}\P{ptcpt+1}\V{i+1}"    

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            print(f"Processing participant {ptcpt+1}, video {i+1}")

            cap = cv2.VideoCapture(vid_path)
            count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_filename = os.path.join(out_dir, f"frame_{count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                count += 1

            cap.release()

        print(f"Done with participant {ptcpt+1}")
       

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("-v", "--video_path", type=str, required=True, help="Path to the input video")
    parser.add_argument("-d", "--output_dir", type=str, required=True, help="Directory to save extracted frames")
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output_dir)


if __name__ == "__main__":
    main()