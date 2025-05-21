import csv
import os

from src.dataloaders.dataset import SubjectEEG
from src.utils.config import CN_NOTCH_MIN, CM_NOTCH_MAX

FS = 250
N_CH = 128
LANZHOU_DATA_DIR = "datasets_cleaned/lanzhou"
LANZHOU_DATA_MAP_FILEPATH = "datasets/lanzhou_map.csv"
SFP_FILEPATH = "datasets/lanzhou.sfp"
LANZHOU_DATASET = {}
LANZHOU_MAP = {}

with open(LANZHOU_DATA_MAP_FILEPATH) as open_file:
    data = open_file.readlines()
    header = data[0]
    data = data[1:]

    r = csv.reader(data)
    for item in r:
        LANZHOU_MAP[item[0]] = item[1]

taskname = "Resting"
LANZHOU_DATASET[taskname] = []
for filename in os.listdir(LANZHOU_DATA_DIR):
    subject_id = filename.split(".")[0]
    filepath = os.path.join(LANZHOU_DATA_DIR, filename)
    group = LANZHOU_MAP[subject_id]

    subject_eeg = SubjectEEG(
        subject_id=subject_id,
        taskname=taskname,
        filepath=filepath,
        group=group,
        fs=FS,
        n_ch=N_CH,
        sfp_filepath=SFP_FILEPATH,
        segment_length=200,
        clip_t=None,
        clip_c=(0, N_CH),
        skip_interpolation=True,
        notch_min=CN_NOTCH_MIN,
        notch_max=CM_NOTCH_MAX,
    )
    LANZHOU_DATASET[taskname].append(subject_eeg)
