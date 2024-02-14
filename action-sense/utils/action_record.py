from .video_record import VideoRecord


class ActionRecord(VideoRecord):
    def __init__(self, tup, dataset_conf):
        self._index = str(tup[0])
        self._series = tup[1]
        self.dataset_conf = dataset_conf

    @property
    def index(self):
        return self._index

    @property
    def file(self):
        return self._series['file']

    @property
    def start_time(self):
        return self._series[
    
    @property
    def start_frame(self):
        return self._series['start_frame'] - 1

    @property
    def end_frame(self):
        return self._series['stop_frame'] - 2

    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame,
                'Flow': int((self.end_frame - self.start_frame) / 2),
                'Event': int((self.end_frame - self.start_frame) / self.dataset_conf["Event"].rgb4e),
                'Spec': self.end_frame - self.start_frame}

    @property
    def label(self):
        if 'verb_class' not in self._series.keys().tolist():
            raise NotImplementedError
        return self._series['verb_class']
