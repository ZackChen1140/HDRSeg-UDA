import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from engine.count_dataloader import RLMDImgAnnDataset
from engine.category import Category, count_categories
from engine.transform import LoadAnn, Resize, LoadAnn

def parse_args():
    parser = argparse.ArgumentParser(
        description="Count pixels per category in a dataset."
    )
    parser.add_argument("csv", type=str, help="The csv file for categories")
    parser.add_argument("label_directory", type=str, help="Directory containing the P-mode png labels.")
    parser.add_argument(
        "--rcs-file-savepath",
        type=str,
        default=None,
        help="If specified, a category statistics file used for class-balanced sampling will be saved.",
    )
    args = parser.parse_args()
    return args


def count_dataset_categories(dataloader, categories, rcs):
    counts = torch.zeros(len(categories)).int()
    for data in tqdm(dataloader, desc="Counting categories..."):
        label = data["ann"]
        count = count_categories(label, categories)
        counts += count

        if rcs is not None:
            rcs.append({"filename": data["ann_path"][0], "count": count.tolist()})
    return counts.tolist()

def main(cate_filepath, img_dirpath, ann_dirpath, rcs_savepath):
    categories = Category.load(cate_filepath, False)

    ann_dataset = RLMDImgAnnDataset(
        img_dir=img_dirpath,
        ann_dir=ann_dirpath,
        transforms=[
            LoadAnn(),
            Resize(),
        ]
    ) 

    dataloader = DataLoader(
        dataset=ann_dataset,
        batch_size=1,
        pin_memory=False,
        shuffle=False,
        num_workers=False,
        drop_last=False
    )

    if rcs_savepath:
        rcs = []
        counts = count_dataset_categories(dataloader, categories, rcs)
        with open(rcs_savepath, "w") as f:
            json.dump(rcs, f)


if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 5
    cate_filepath = sys.argv[1]
    img_dirpath = sys.argv[2]
    ann_dirpath = sys.argv[3]
    rcs_savepath = sys.argv[4]
    main(cate_filepath, img_dirpath, ann_dirpath, rcs_savepath)
