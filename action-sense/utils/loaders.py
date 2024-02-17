import glob
from abc import ABC
import pandas as pd
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger
import numpy as np
from .action_record import ActionRecord


class ActionSenseDataset(data.Dataset, ABC):
    def __init__(
        self,
        modalities,
        mode,
        dataset_conf,
        mode_config,
        num_clips,
        transform=None,
        load_feat=False,
        additional_info=False,
        multimodal=False,
        **kwargs,
    ):
        """
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = (
            modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        )
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.additional_info = additional_info
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat
        self.mode_config = mode_config
        self.num_clips = num_clips

        if len(self.modalities) > 1 or self.modalities[0] == "RGB" or multimodal:
            pickle_name = f"{self.mode}_MULTIMODAL"
        else:
            pickle_name = f"{self.mode}_{self.modalities[0]}"

        pickle_name += ".pkl"

        logger.info(f"Loading data from {pickle_name}.")

        self.data_path = os.path.join(
            self.dataset_conf["annotations_path"], pickle_name
        )

        data = pd.read_pickle(self.data_path)

        """ record_list = [
            ActionRecord(info, self.dataset_conf) for info in data.iterrows()
        ] """
        self.record_list = tuple(
            ActionRecord(info, self.dataset_conf) for info in data.iterrows()
        )

        self.features = {}
        for m in self.modalities:
            if self.load_feat[m]:
                self.features[m] = pd.DataFrame(
                    pd.read_pickle(
                        os.path.join(
                            "saved_features",
                            pickle_name,
                        )
                    )["features"]
                )

    def _getEMG(self, index):
        record = self.record_list[index]
        sides = ["left", "right"]
        emgs = []
        for s in sides:
            modality = "EMG"
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices = self._get_train_indices(modality, record, side=s)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices = self._get_val_indices(modality, record, side=s)
            if s == "left":
                emg = record.myo_left_readings
            else:
                emg = record.myo_right_readings
            emg = emg[segment_indices, :]
            emgs.append(emg)
        # Concatenate along the second axis to create a 100x16 array
        result = np.concatenate((emgs[0], emgs[1]), axis=1, dtype=np.float32)
        if self.transform["EMG"] is not None:
            result = self.transform["EMG"](result)
        return result, record.label

    def __getitem__(self, index):
        item = {}
        for m in self.modalities:
            if m == "EMG":
                item[m], label = self._getEMG(index)
            else:
                item[m], label = self._get_RGB(index)
        if self.additional_info:
            return item, label, index
        else:
            return item, label

    def _get_train_indices(self, modality, record: ActionRecord, side=None):
        if side is not None:
            submodality = modality + "_" + side
        else:
            submodality = modality

        def random_offset(highest_idx):
            if highest_idx == 0:
                offset = 0
            else:
                offset = np.random.randint(0, highest_idx + 1)
            return offset

        frame_idx = []
        # Dense sampling
        if self.mode_config.dense_sampling[modality]:
            # Finds the highest possible index
            highest_idx = max(
                0,
                record.num_frames(submodality)
                - (self.mode_config.stride[modality])
                * self.mode_config.num_frames_per_clip[modality],
            )
            # Selects one random initial frame for each clip
            for _ in range(self.num_clips):
                # Selects a random offset between 0 and the highest possible index
                offset = random_offset(highest_idx)
                # Selects the frames for the clip
                frames = [
                    (offset + self.mode_config.stride[modality] * x)
                    % (record.num_frames(submodality) - 1)
                    for x in range(self.mode_config.num_frames_per_clip[modality])
                ]
                # Appends the frames to the clips list
                frame_idx.extend(frames)
        # Uniform sampling
        else:
            # Frames available for each clip
            clip_frames = max(
                record.num_frames(submodality) // self.num_clips,
                self.mode_config.num_frames_per_clip[modality],
            )
            clip_frames = min(clip_frames, record.num_frames(submodality))
            highest_idx = max(0, record.num_frames(submodality) - clip_frames)
            for _ in range(self.num_clips):
                # Selects a random offset between 0 and the highest possible index
                offset = random_offset(highest_idx)
                # Selects K evenly spaced frames from each clip
                frames = np.linspace(
                    offset,
                    offset + clip_frames - 1,
                    self.mode_config.num_frames_per_clip[modality],
                    dtype=int,
                )
                frame_idx.extend(frames)
        frame_idx = np.asarray(frame_idx)
        return frame_idx

    def _get_val_indices(self, modality, record: ActionRecord, side=None):
        if side is not None:
            submodality = modality + "_" + side
        else:
            submodality = modality

        frame_idx = []
        # Dense sampling
        if self.mode_config.dense_sampling[modality]:
            # Number of frames in each half of the clip
            clip_frames = record.num_frames(submodality) // self.num_clips // 2
            # Space between first and last frame taken from each clip
            tot_frames = (
                self.mode_config.num_frames_per_clip[modality]
                * self.mode_config.stride[modality]
            )
            if tot_frames // 2 <= clip_frames:
                # Selects evenly spaced center points for each clip
                center_points = np.linspace(
                    clip_frames,
                    record.num_frames(submodality) - clip_frames,
                    self.num_clips,
                    dtype=int,
                )
            else:
                # Some segments overlap
                center_points = np.linspace(
                    tot_frames // 2,
                    record.num_frames(submodality) - tot_frames // 2,
                    self.num_clips,
                    dtype=int,
                )
            # Selects the frames for each clip
            for center in center_points:
                frames = [
                    (center - tot_frames // 2 + self.mode_config.stride[modality] * x)
                    % (record.num_frames(submodality) - 1)
                    for x in range(self.mode_config.num_frames_per_clip[modality])
                ]
                frame_idx.extend(frames)
        # Uniform sampling
        else:
            frame_idx = np.linspace(
                0,
                record.num_frames(submodality) - 1,
                self.mode_config.num_frames_per_clip[modality] * self.num_clips,
                dtype=int,
            )
        return np.asarray(frame_idx)

    def _get_RGB(self, index):
        """Only for RGB modality, for now."""
        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.record_list[index]

        if self.load_feat:
            sample_row = self.features["RGB"][
                self.features["RGB"]["uid"] == int(record.uid)
            ]
            assert len(sample_row) == 1
            sample = sample_row["features_" + "RGB"].values[0]

            return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        if self.mode == "train":
            # here the training indexes are obtained with some randomization
            segment_indices = self._get_train_indices("RGB", record)
        else:
            # here the testing indexes are obtained with no randomization, i.e., centered
            segment_indices = self._get_val_indices("RGB", record)

        img, label = self.get("RGB", record, segment_indices)
        frames = img

        return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_image(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        if self.transform[modality] is not None:
            images = self.transform[modality](images)
        return images, record.label

    def _load_image(self, modality, record: ActionRecord, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == "RGB" or modality == "RGBDiff":
            # here the offset for the starting index of the sample is added
            idx_untrimmed = record.start_frame[modality] + idx
            try:
                img = Image.open(
                    os.path.join(
                        data_path,
                        tmpl.format(idx_untrimmed),
                    )
                ).convert("RGB")
            except FileNotFoundError:
                print(
                    "Img not found:",
                    os.path.join(
                        data_path,
                        tmpl.format(idx_untrimmed),
                    ),
                )
                max_idx_video = int(
                    sorted(
                        glob.glob(
                            os.path.join(
                                data_path, record.untrimmed_video_name, "img_*"
                            )
                        )
                    )[-1]
                    .split("_")[-1]
                    .split(".")[0]
                )
                if idx_untrimmed > max_idx_video:
                    img = Image.open(
                        os.path.join(
                            data_path,
                            record.untrimmed_video_name,
                            tmpl.format(max_idx_video),
                        )
                    ).convert("RGB")
                else:
                    raise FileNotFoundError
            return [img]

        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.record_list)
