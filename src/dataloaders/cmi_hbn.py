import glob
import os

from src.dataloaders.dataset import SubjectEEG

FS = 500
N_CH = 128
SFP_FILEPATH = "datasets/cmi_hbn_GSN_HydroCel_129.sfp"
SUBJECT_MAX_T = {
    "NDARAD459XJK": (0, 120000),
    "NDARAG429CGW": (0, 160000),
    "NDARBH789CUP": (11000, 150000),
    "NDARAD459XJK": (45000, 110000),
}

EEG_TASK_MAP = {
    "Resting": "RestingState_data.csv",
    "SAIIT2AFC": "SAIIT_2AFC_Block1_data.csv",
    "SurroundSuppression": "SurroundSupp_Block1_data.csv",
    "VideoDecisionMaking": "Video-DM_data.csv",
    "VideoFeedForward": "Video-FF_data.csv",
    "VideoTemporalPrediction": "Video-TP_data.csv",
    "VideoWorkingMemory": "Video-WK_data.csv",
    "VisualLearning": "vis_learn_data.csv",
    "WISCProcessingSpeed": "WISC_ProcSpeed_data.csv",
}

CMI_HBN_DATASET = {}

for taskname, filename in EEG_TASK_MAP.items():
    dir_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    search_pattern = os.path.join(
        "datasets_cleaned", "cmi_hbn", "*", "EEG", "raw", "csv_format", filename
    )
    filepaths = glob.glob(search_pattern)
    for filepath in filepaths:
        subject_id = filepath.split(os.path.sep)[2]
        if taskname not in CMI_HBN_DATASET:
            CMI_HBN_DATASET[taskname] = []

        clip_t = SUBJECT_MAX_T.get(subject_id, None)

        subject_eeg = SubjectEEG(
            subject_id=subject_id,
            taskname=taskname,
            filepath=filepath,
            group="UNKNOWN",
            fs=FS,
            n_ch=N_CH,
            sfp_filepath=SFP_FILEPATH,
            segment_length=200,
            clip_t=clip_t,
            skip_interpolation=False,
        )
        CMI_HBN_DATASET[taskname].append(subject_eeg)

# def load_with_preprocessing(
#     filepath: str,
#     subject_id: str,
#     *,
#     n_ch: int = 128,
#     max_t: int = -1,
#     skip_znorm: bool = False,
#     skip_interpolation: bool = False,
#     fs: int = FS,
# ) -> Any:
#     X = np.loadtxt(filepath, delimiter=",")
#     if subject_id in SUBJECT_MAX_T:
#         s_min_t, s_max_t = SUBJECT_MAX_T[subject_id]
#         X = X[:n_ch, s_min_t:s_max_t]
#     X = X[:n_ch, :max_t]
#     X = fill_flat_channels(X, fillval=np.nan)
#     X = fill_wack_channels(X, fillval=np.nan)
#     if not skip_interpolation:
#         X = interpolate_faulty_channels(X, "GSN_HydroCel_129.sfp", fs=fs)
#     X = butter_bandpass_filter(X, lowcut=BP_MIN, highcut=BP_MAX, fs=fs)
#     X = butter_bandstop_filter(X, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=fs)
#     if not skip_znorm:
#         X = znorm(X)
#     return X


# # def collect_specified_files(taskname: str) -> List[str]:

# #     filename = EEG_TASK_MAP[taskname]
# #     dir_path = os.path.dirname(
# #         os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# #     )
# #     search_pattern = os.path.join(
# #         dir_path, "data", "*", "EEG", "raw", "csv_format", filename
# #     )

# #     file_paths = glob.glob(search_pattern)
# #     print(f"Collected {len(file_paths)} files for {taskname}")
# #     return file_paths
