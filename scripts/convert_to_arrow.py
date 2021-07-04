from argparse import ArgumentParser

import pyarrow as pa
from pyarrow import json

import datasets as nlp


def convert_to_arrow(
    file_paths, save_path, cache_path_prefix="./data_chunk", no_combine=False
):
    converted_tables = []

    if len(file_paths) == 1:
        mmap = pa.memory_map(file_paths[0])
        json_input = json.read_json(mmap)
        writer = nlp.arrow_writer.ArrowWriter(path=save_path)
        writer.write_table(json_input)
    else:

        for idx, file in enumerate(file_paths):
            cache_path = cache_path_prefix + "." + str(idx)
            mmap = pa.memory_map(file)
            json_input = json.read_json(mmap)
            writer = nlp.arrow_writer.ArrowWriter(path=cache_path)
            writer.write_table(json_input)

            mmap = pa.memory_map(cache_path)
            f = pa.ipc.open_stream(mmap)
            pa_table = f.read_all()

            converted_tables.append(pa_table)

        if not no_combine:
            pa_table = pa.concat_tables(converted_tables, promote=False)

            writer = nlp.arrow_writer.ArrowWriter(path=save_path)
            writer.write_table(pa_table)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--file_paths",
        type=str,
        nargs="+",
        help="The paths to the JSON files to convert to arrow and combine.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data.arrow",
        help="The path to save the combined arrow file to. Defaults to './data.arrow'.",
    )
    parser.add_argument(
        "--cache_path_prefix",
        type=str,
        default="./data_chunk",
        help="The cache path and file name prefix for the converted JSON files. "
        + "Defaults to './data_chunk'.",
    )
    parser.add_argument(
        "--no_combine",
        action="store_true",
        help="Don't combine the converted JSON files.",
    )

    args = parser.parse_args()

    convert_to_arrow(**vars(args))
