#!/usr/bin/env python

import argparse
import os
import sys

import toml


def parse_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        metavar="FILE",
        default="pyproject.toml",
        help="input pyproject.toml",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        metavar="FILE",
        default="requirements.txt",
        help="output requirements.txt",
    )
    return parser.parse_args()


def main() -> int:
    opts = parse_argv()
    with open(opts.input, mode="r", encoding="utf-8") as f:
        data = toml.loads(f.read())

    with open(opts.output, mode="w", encoding="utf-8") as f:
        f.write(
            """\
# AUTO GENERATED FILE, DO NOT EDIT
# Run `make requirements.txt` to generate this file

"""
        )
        f.write("# project.dependencies\n")
        for package in data["project"]["dependencies"]:
            f.write(f"{package}\n")

        for dep in data["project"]["optional-dependencies"]:
            f.write(f"\n# project.dependencies.{dep}\n")
            for package in data["project"]["optional-dependencies"][dep]:
                f.write(f"{package}\n")

    print(f"Written to {os.path.abspath(opts.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
