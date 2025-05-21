import os

from src.dataloaders.dataset import SubjectEEG

FS = 250
N_CH = 19
IBIB_PAN_DATA_DIR = "datasets_cleaned/ibib_pan"
SFP_FILEPATH = "datasets/ibib_pan.sfp"
IBIB_PAN_DATASET = {}
taskname = "Resting"
IBIB_PAN_DATASET[taskname] = []

for subject_id in os.listdir(IBIB_PAN_DATA_DIR):
    filepath = os.path.join(IBIB_PAN_DATA_DIR, subject_id)
    if not filepath.endswith(".csv"):
        continue
    group = "SCHIZO" if subject_id.startswith("s") else "HEALTHY"

    subject_eeg = SubjectEEG(
        subject_id=subject_id,
        taskname=taskname,
        filepath=filepath,
        group=group,
        fs=FS,
        n_ch=N_CH,
        sfp_filepath=SFP_FILEPATH,
        segment_length=25,
        clip_t=None,
        skip_interpolation=True,
    )
    IBIB_PAN_DATASET[taskname].append(subject_eeg)
