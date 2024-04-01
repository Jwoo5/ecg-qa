import os
import glob
import argparse
import json
import shutil
import pandas as pd
from pathlib import Path
import subprocess
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", default=".",
        help='root directory containing ecgqa files (e.g., ecgqa/mimic-iv-ecg/)'
    )
    parser.add_argument(
        "--mimic-iv-ecg-data-dir", metavar="DIR", default=None,
        help="directory containing mimic-iv-ecg data "
            "($mimic_iv_ecg_data_dir/files/pNNNN/pXXXXXXXX/sZZZZZZZZ/ZZZZZZZZ.dat)"
    )
    parser.add_argument(
        "--dest", type=str, metavar="DIR", default="output/mimic-iv-ecg",
        help='output directory'
    )

    return parser

def main(args):
    dir_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)
    subdirs = ["template", "paraphrased"]

    cache_path = os.path.join(str(Path.home()), ".cache/ecgqa")
    
    if args.mimic_iv_ecg_data_dir is None:
        if not os.path.exists(os.path.join(cache_path, "mimic-iv-ecg")):
            os.makedirs(os.path.join(cache_path, "mimic-iv-ecg"))
            subprocess.run([
                "wget", "-r", "-N", "-c", "np",
                "https://physionet.org/files/mimic-iv-ecg/1.0/"
                "-P", os.path.join(cache_path, "mimic-iv-ecg")
            ])
            shutil.move(
                os.path.join(cache_path, "mimic-iv-ecg", "physionet.org/files/mimic-iv-ecg/1.0/files"),
                os.path.join(cache_path, "mimic-iv-ecg")
            )
            shutil.rmtree(os.path.join(cache_path, "physionet.org"))
            
        mimic_iv_ecg_data_dir = os.path.join(cache_path, "mimic-iv-ecg")
    else:
        mimic_iv_ecg_data_dir = args.mimic_iv_ecg_data_dir
    record_list = pd.read_csv(os.path.join(mimic_iv_ecg_data_dir, "record_list.csv"))
    record_list = record_list.set_index("study_id")

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
        if dest_path != dir_path:
            shutil.copy2(
                os.path.join(dir_path, "answers.csv"), os.path.join(dest_path, "answers.csv")
            )
            shutil.copy2(
                os.path.join(dir_path, "answers_for_each_template.csv"),
                os.path.join(dest_path, "answers_for_each_template.csv")
            )

    for subdir in subdirs:
        for fname in glob.iglob(os.path.join(dir_path, subdir, "**/*.json")):
            split = fname.split("/")[-2]
            basename = os.path.basename(fname)
            if not os.path.exists(os.path.join(dest_path, subdir, split)):
                os.makedirs(os.path.join(dest_path, subdir, split))

            with open(fname, "r") as f:
                data = json.load(f)

            for i, sample in tqdm(enumerate(data), total=len(data), desc=os.path.join(subdir, split, basename)):
                sample["ecg_path"] = []
                for ecg_id in sample["ecg_id"]:
                    data_path = os.path.join(mimic_iv_ecg_data_dir, record_list.loc[ecg_id]["path"])
                    if not os.path.exists(data_path + ".dat"):
                        raise FileNotFoundError(
                            f"{data_path}",
                            "If you ran the script without --mimic-iv-ecg-data-dir, it may mean that "
                            "the download has been failed by an unknown error. Please run the script "
                            f"again after removing {cache_path}/mimic-iv-ecg."
                        )
                    sample["ecg_path"].append(data_path)
            with open(os.path.join(dest_path, subdir, split, basename), "w") as f:
                json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)