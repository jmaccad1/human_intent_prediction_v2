import argparse
import sys
from os import path as osp

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DETECTRON2_ROOT = osp.join(PROJECT_ROOT, "src", "detectron2")

for import_root in (DETECTRON2_ROOT, PROJECT_ROOT):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import pandas as pd

PATH = "your_folder_path"
START_PTCPT = 1
GAZE_INOUT_COL = 5

def get_metric(metric_idx, ROOT):
    overall_metric = []
    sample = 0
    x = metric_idx # 1 for auc, 2 for m_dist, 3 for ave_dist
    
    if x == 1:
        metric_name = "AUC"
    elif x == 2:
        metric_name = "m_dist"
    elif x == 3:
        metric_name = "ave_dist"

    for ptcpt in range(START_PTCPT-1, 1):
        ptcpt_metric = []
        for i in range(50):
            
            root = f"{PATH}{ROOT}/"

            try:
                df = pd.read_csv(f"{root}/metrics_p{ptcpt+1}_v{i+1}.txt", header=None)
                if GAZE_INOUT_COL in df.columns:
                    df = df[df[GAZE_INOUT_COL] == 1]
                else:
                    print(
                        f"metrics_p{ptcpt+1}_v{i+1}.txt has no gaze_inout column; "
                        "using all rows."
                    )

                metric_values = df[x].dropna().tolist()
                overall_metric.extend(metric_values)
                ptcpt_metric.extend(metric_values)
                sample += len(metric_values)
                #print(sample)
                #print(f"P{ptcpt+1}_V{i+1} {metric_name}: {df[x].mean()} | Std: {df[x].std()}")

            except FileNotFoundError:
                print(f"Metrics file not found for P{ptcpt+1}_V{i+1}, skipping.")
        if ptcpt_metric:
            ptcpt_metric = pd.Series(ptcpt_metric)
            print(f"P{ptcpt+1}: Average {metric_name}: {ptcpt_metric.mean()} | Std: {ptcpt_metric.std()}")
        else:
            print(f"P{ptcpt+1}: No {metric_name} values found.")

    if len(overall_metric) == sample:
        overall_metric = pd.Series(overall_metric)
        print(f"Average {metric_name}: {overall_metric.mean()} | Std: {overall_metric.std()}")
        #print(f"Total samples: {sample}")
    else:
        print(f"Error: Number of {metric_name} values collected ({len(overall_metric)}) does not match expected sample size ({sample}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average metrics across participants and videos.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--root", type=str, default=None, help="Metrics output folder, e.g. infer_out_05_04.")
    args = parser.parse_args() 

    if args.root is not None:
        root = "\\" + args.root.strip("\\/")
    elif args.model == "gazefollow":
        root = r"\infer_out_05_12"
    elif args.model == "videoattentiontarget":
        root = r"\infer_out_05_121"
    else:
        raise ValueError(f"Unknown model '{args.model}'. Use 'gazefollow' or 'videoattentiontarget'.")

    get_metric(1, root)
    #get_metric(2, root)
    #get_metric(3, root)
    #get_mdist()
    #get_avedist()
