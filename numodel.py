import torch as th
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import os

from s3dg import Sentence_Embedding


class Model(nn.Module):
    '''
    Video model: https://pytorch.org/docs/stable/torchvision/models.html#video-classification

    extra args kept for easy compatibility
    '''
    def __init__(self, num_classes=512, gating=True, space_to_depth=False,
                  word2vec_path='', init='uniform', token_to_word_path='data/dict.npy'):
        super(Model, self).__init__()
        self.video_model = torchvision.models.video.r2plus1d_18(pretrained=True)
        self.video_model.fc = nn.Linear(512, num_classes)

        self.text_module = Sentence_Embedding(
                               num_classes,
                               os.path.join(os.path.dirname(__file__), token_to_word_path),
                               word2vec_path=os.path.join(os.path.dirname(__file__), word2vec_path))

    def forward(self, video, text, mode='all', mixed5c=False):
        video, text = self.preprocess(video, text)
        if mode == 'all':
            return self.video_model(video), self.text_module(text)
        elif mode == 'video':
            return self.video_model(video)
        elif mode == 'text':
            return self.text_module(text)
        else:
            raise NotImplementedError

    def preprocess(self, video, text):
        video = self.normalize_video(video)
        # preprocess text??
        return video, text

    def normalize_video(self, video):
        # TODO: normalize according to https://pytorch.org/docs/stable/torchvision/models.html#video-classification
        return video

        

model = Model()
print(model)
