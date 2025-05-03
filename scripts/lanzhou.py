import os
import re
from pathlib import Path

import pandas as pd
import scipy.io


def convert_mat_to_csv(raw_dir: str, output_dir: str) -> None:
    map_file = "data_raw/lanzhou/map.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    map_df = pd.read_csv(map_file, converters={"subject id": str})
    valid_subjects = set(map_df["subject id"].astype(str))

    pattern = re.compile(r"^(\d+).*\.mat$")

    for filename in os.listdir(raw_dir):
        if not filename.endswith(".mat"):
            continue

        match = pattern.match(filename)
        if not match:
            continue

        subject_id = match.group(1)
        if subject_id not in valid_subjects:
            continue
        mat_path = os.path.join(raw_dir, filename)
        mat_data = scipy.io.loadmat(mat_path)

        data_key = [k for k in mat_data.keys() if not k.startswith("__")][0]
        data = mat_data[data_key]
        df = pd.DataFrame(data)

        output_path = os.path.join(output_dir, f"{subject_id}.csv")
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    raw_dir = "data_raw/lanzhou"
    output_dir = "data/lanzhou"

    convert_mat_to_csv(raw_dir, output_dir)
