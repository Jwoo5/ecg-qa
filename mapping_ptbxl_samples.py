import os
import glob
import argparse
import json
import shutil
from pathlib import Path
import subprocess
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", default=".",
        help='root directory containing ecgqa files (e.g., ecgqa/ptbxl/)'
    )
    parser.add_argument(
        '--ptbxl-data-dir', metavar="DIR", default=None,
        help='directory containing ptbxl data ($ptbxl_data_dir/records500/**/*.mat)'
    )
    parser.add_argument(
        "--dest", type=str, metavar="DIR", default="output/ptbxl",
        help='output directory'
    )

    return parser

def main(args):
    dir_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)
    subdirs = ["template", "paraphrased"]

    cache_path = os.path.join(str(Path.home()), ".cache/ecgqa")
    
    if args.ptbxl_data_dir is None:
        if not os.path.exists(os.path.join(cache_path, "ptbxl")):
            os.makedirs(os.path.join(cache_path, "ptbxl"))
            subprocess.run([
                "wget", "-r", "-N", "-c", "np",
                "https://physionet.org/files/ptb-xl/1.0.3/records500/",
                "-P", os.path.join(cache_path, "ptbxl")
            ])
            shutil.move(
                os.path.join(cache_path, "ptbxl", "physionet.org/files/ptb-xl/1.0.3/records500"),
                os.path.join(cache_path, "ptbxl")
            )
            shutil.rmtree(os.path.join(cache_path, "physionet.org"))
            
        ptbxl_data_dir = os.path.join(cache_path, "ptbxl")
    else:
        ptbxl_data_dir = args.ptbxl_data_dir
    
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
                    if not os.path.exists(os.path.join(get_ptbxl_data_path(ecg_id, ptbxl_data_dir)) + ".dat"):
                        raise FileNotFoundError(
                            os.path.join(get_ptbxl_data_path(ecg_id, ptbxl_data_dir)),
                            "If you ran the script without --ptbxl-data-dir, it may mean that "
                            "the download has been failed by an unknown error. Please run the script "
                            f"again after removing {cache_path}/ptbxl."
                        )
                    sample["ecg_path"].append(get_ptbxl_data_path(ecg_id, ptbxl_data_dir))

            with open(os.path.join(dest_path, subdir, split, basename), "w") as f:
                json.dump(data, f, indent=4)

def get_ptbxl_data_path(ecg_id, ptbxl_data_dir):
    return os.path.join(
        ptbxl_data_dir,
        "records500",
        f"{int(ecg_id / 1000) * 1000 :05d}",
        f"{ecg_id:05d}_hr"
    )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)