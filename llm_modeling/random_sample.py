import os
import argparse
import random
import itertools
import scipy.io

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", default=".",
        help="root directory containing ptbxl files to pre-process"
    )
    parser.add_argument(
        '--subset', type=str, required=True,
        help="name of the subset to be random sampled"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed"
    )

    return parser

def main(args):
    random.seed(args.seed)

    qtypes1 = ["single", "comparison_consecutive", "comparison_irrelevant"]
    qtypes2 = ["verify", "choose", "query"]
    attrs = ["scp-codes", "noise", "infarction_stadium", "extra-systole", "heart-axis", "numeric"]
    qtypes = {i: "-".join(x) for i, x in enumerate(itertools.product(qtypes1, qtypes2, attrs))}
    counts = {key: 0 for key in qtypes.values()}

    with open(os.path.join(args.root, args.subset + ".tsv"), "r") as f:
        root = f.readline().strip()
        for item in f.readlines():
            fname = item.split("\t")[0].strip()
            data = scipy.io.loadmat(os.path.join(root, fname))
            
            key = qtypes[data["question_type3"][0][0]]
            counts[key] += 1

    nsamples_qtypes = {
        k: round(v / 10) for k, v in counts.items()
    }
    qtypes_candidates = {k: [] for k in nsamples_qtypes.keys()}

    sizes = dict()
    with open(os.path.join(args.root, args.subset + ".tsv"), "r") as f:
        root = f.readline().strip()
        for line in f.readlines():
            items = line.strip().split("\t")
            fname = items[0]
            ecg_size = items[1]
            text_size = items[-1]
            if len(items) == 4:
                ecg_size_2 = items[2]
                sizes[fname] = (ecg_size, ecg_size_2, text_size)
            else:
                sizes[fname] = (ecg_size, text_size)
            data = scipy.io.loadmat(os.path.join(root, fname))
            key = qtypes[data["question_type3"][0][0]]
            qtypes_candidates[key].append(fname)

    qtypes_sampled = {
        k: random.sample(v, nsamples_qtypes[k]) for k, v in qtypes_candidates.items()
    }

    with open(os.path.join(args.root, args.subset + "_sampled.tsv"), "w") as f:
        print(root, file=f)
        for fnames in qtypes_sampled.values():
            for fname in fnames:
                print(fname, end="\t", file=f)
                print(sizes[fname][0], end="\t", file=f)
                if len(sizes[fname]) == 3:
                    print(sizes[fname][1], end="\t", file=f)
                    print(sizes[fname][2], file=f)
                else:
                    print(sizes[fname][1], file=f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)