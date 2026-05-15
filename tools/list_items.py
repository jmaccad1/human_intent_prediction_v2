import argparse
import os


def count_files(path):
    total = 0
    for root, dirs, files in os.walk(path):
        total += len(files)
    return total

def count_hiphop(path):
    total = 0
    for i in range(20):
        video_path = os.path.join(path, f"P{i+1}")
        for j in range(50):
            video_path_j = os.path.join(video_path, f"V{j+1}")
            if os.path.isdir(video_path_j):
                total += count_files(video_path_j)
            else:
                print(f"Directory {video_path_j} does not exist, skipping.")

    return total

def main():
    parser = argparse.ArgumentParser(description="Count total number of files in a directory.")
    parser.add_argument("-p", "--path", help="Path to the directory to count files in")
    parser.add_argument("-sf", "--skip_frames", type=int, default=4, help="Number of frames to skip (default: 4)")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise SystemExit(f"Error: '{args.path}' is not a valid directory")


    print(count_hiphop(args.path)*0.8/args.skip_frames)


if __name__ == "__main__":
    main()
