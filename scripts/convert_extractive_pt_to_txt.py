import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm


def convert_extractive_pt_to_txt(path):
    full_directory_listing = os.listdir(path)
    all_pt_files = [
        x for x in full_directory_listing if os.path.splitext(x)[1] == ".pt"
    ]
    for file_path in tqdm(all_pt_files, desc="Converting PT to TXT"):
        file_path = os.path.join(path, file_path)
        torch_data = torch.load(file_path)
        with open(file_path[:-2] + "txt", "w+") as file:
            file.write("\n".join([str(x).replace("'", '"') for x in torch_data]) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        help="The path to the '.pt' files to convert to text.",
    )

    args = parser.parse_args()

    convert_extractive_pt_to_txt(**vars(args))
