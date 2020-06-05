
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import pickle


class Youcook_DataLoader(Dataset):
    """Youcook Video-Text loader."""

    def __init__(
            self,
            data,
            video_root='',
            num_clip=4,
            fps=16,
            num_frames=32,
            size=224,
            crop_only=False,
            center_crop=True,
            token_to_word_path='data/dict.npy',
            max_words=30,
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.data = pd.read_csv(data)
        self.video_root = video_root
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_clip = num_clip
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.max_words = max_words
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def __len__(self):
        return len(self.data)

    def _get_video(self, video_path, start, end, num_clip):
        video = th.zeros(num_clip, 3, self.num_frames, self.size, self.size)
        start_ind = np.linspace(start, max(start, end-self.num_sec - 0.4), num_clip) 
        for i, s in enumerate(start_ind):
            video[i] = self._get_video_start(video_path, s) 
        return video

    def _get_video_start(self, video_path, start):
        start_seek = start
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        return video[:, :self.num_frames]

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        task = self.data['task'].values[idx]
        start = self.data['start'].values[idx]
        end = self.data['end'].values[idx]
        cap = self.data['text'].values[idx]
        if os.path.isfile(os.path.join(self.video_root, 'validation', str(task), video_id + '.mp4')):
            video_path = os.path.join(self.video_root, 'validation', str(task), video_id + '.mp4')
        elif os.path.isfile(os.path.join(self.video_root, 'validation', str(task), video_id + '.mkv')):
            video_path = os.path.join(self.video_root, 'validation', str(task), video_id + '.mkv')
        elif os.path.isfile(os.path.join(self.video_root, 'validation', str(task), video_id + '.webm')):
            video_path = os.path.join(self.video_root, 'validation', str(task), video_id + '.webm')
        else:
            raise ValueError
        text = self.words_to_ids(cap)
        video = self._get_video(video_path, start, end, self.num_clip)
        return {'video': video, 'text': text}

