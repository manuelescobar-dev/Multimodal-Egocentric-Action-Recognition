def get_train_indices(self, record: EpicVideoRecord = "RGB"):
    def random_offset(highest_idx):
        if highest_idx == 0:
            offset = 0
        else:
            offset = np.random.randint(0, highest_idx)
        return offset

    clips = []
    # Dense sampling
    if self.dense_sampling:
        # Finds the highest possible index
        highest_idx = max(
            0,
            record.num_frames - (self.stride) * self.num_frames_per_clip,
        )
        # Selects one random initial frame for each clip
        for _ in range(self.num_clips):
            # Selects a random offset between 0 and the highest possible index
            offset = random_offset(highest_idx)
            # Selects the frames for the clip
            frames = [
                (offset + self.stride * x) for x in range(self.num_frames_per_clip)
            ]
            # Appends the frames to the clips list
            clips.append(frames)
    # Uniform sampling
    else:
        # Frames available for each clip
        clip_frames = record.num_frames // self.num_clips
        highest_idx = max(0, record.num_frames - self.num_frames_per_clip)
        for _ in range(self.num_clips):
            # Selects a random offset between 0 and the highest possible index
            offset = random_offset(highest_idx)
            # Selects K evenly spaced frames from each clip
            frames = np.linspace(
                offset,
                offset + clip_frames,
                self.num_frames_per_clip,
                dtype=int,
            )
            clips.append(frames)
    frame_idx = np.array(clips)
    return frame_idx


def get_val_indices(self, record: EpicVideoRecord):
    clips = []
    # Dense sampling
    if self.dense_sampling:
        tot_frames = (self.num_frames_per_clip * self.stride) // 2
        offsets = np.linspace(
            tot_frames,
            record.num_frames - tot_frames,
            self.num_clips + 1,
            dtype=int,
        )[1:]
    # Uniform sampling
    else:
        return np.linspace(
            0,
            record.num_frames - 1,
            self.num_frames_per_clip * self.num_clips,
        ).reshape(self.num_clips, self.num_frames_per_clip)
