import argparse
import json
import os
import warnings

from prettify_data import format_data


def add_rel(args: argparse.Namespace) -> None:
    if args.output is None:  # Overwrite input data file if output path is not provided.
        args.output = args.dst
    with open(args.src, "rt") as src_file:
        dst_json = (
            json.load(open(args.dst, "rt")) if os.path.exists(args.dst) else dict()
        )
        src_json = json.load(src_file)
        for rel in args.rels:
            if rel not in src_json:
                warnings.warn(
                    "Relation [{}] is not in source data file [{}]".format(
                        rel, args.src
                    )
                )
                continue
            if rel in dst_json:
                warnings.warn(
                    "Relation [{}] is already in destination data file [{}] "
                    "Augmenting the existing examples from the source data file [{}]".format(
                        rel, args.dst, args.src
                    )
                )
                dst_json[rel].extend(src_json[rel])
            else:
                dst_json[rel] = src_json[rel]
    with open(args.output, "wt") as out_file:
        out_file.write(format_data(dst_json))


def del_rel(args: argparse.Namespace) -> None:
    if args.output is None:  # Overwrite input data file if output path is not provided.
        args.output = args.input

    with open(args.input, "rt") as in_file:
        in_json = json.load(in_file)
        for rel in args.rels:
            if rel not in in_json:
                warnings.warn(
                    "Relation [{}] is not in data file [{}]. Skipping.".format(
                        rel, args.input
                    )
                )
            else:
                del in_json[rel]
    with open(args.output, "wt") as out_file:
        out_file.write(format_data(in_json))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Functions: choose from 'add_rel' or 'del_rel'"
    )

    # Parser for adding relations.
    parser1 = subparsers.add_parser(
        "add_rel", help="Add relation(s) to a dataset from another dataset."
    )
    parser1.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source data file containing relation(s) to be added.",
    )
    parser1.add_argument(
        "--dst", type=str, required=True, help="Destination data file."
    )
    parser1.add_argument("--output", type=str, help="Output path.")
    parser1.add_argument("--rels", nargs="*", help="Relations to be added.")
    parser1.set_defaults(func=add_rel)

    # Parser for removing relations.
    parser2 = subparsers.add_parser(
        "del_rel", help="Remove relation(s) from a dataset."
    )
    parser2.add_argument(
        "--input", type=str, required=True, help="Path to the input data file."
    )
    parser2.add_argument("--output", type=str, help="Output path.")
    parser2.add_argument("--rels", nargs="*", help="Relations to be removed.")
    parser2.set_defaults(func=del_rel)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
