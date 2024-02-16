from .video_record import VideoRecord


class ActionRecord(VideoRecord):
    def __init__(self, data, dataset_conf):
        self._data = data[1]
        self.dataset_conf = dataset_conf

    @property
    def myo_left_timestamps(self):
        return self._data["myo_left_timestamps"]

    @property
    def myo_right_timestamps(self):
        return self._data["myo_right_timestamps"]

    @property
    def myo_left_readings(self):
        return self._data["myo_left_readings"]

    @property
    def myo_right_readings(self):
        return self._data["myo_right_readings"]

    @property
    def start_frame(self):
        return {
            "RGB": self._data["start_frame"],
            "EMG_left": self.myo_left_timestamps[0],
            "EMG_right": self.myo_right_timestamps[0],
        }

    @property
    def end_frame(self):
        return {
            "RGB": self._data["stop_frame"],
            "EMG_left": self.myo_left_timestamps[-1],
            "EMG_right": self.myo_right_timestamps[-1],
        }

    def num_frames(self, submodality):
        frames = {
            "RGB": self.end_frame["RGB"] - self.start_frame["RGB"],
            "EMG_left": len(self.myo_left_timestamps),
            "EMG_right": len(self.myo_right_timestamps),
        }
        return frames[submodality]

    @property
    def label(self):
        return self._data["label"]
