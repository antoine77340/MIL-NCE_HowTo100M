import os
import random
import time

import torch
import torch.utils.data

from metrics import compute_metrics, print_computed_metrics
from args import get_args
from msrvtt_loader import MSRVTT_DataLoader
import s3dg
from tqdm import tqdm
import numpy as np


def main():
    args = get_args()
    assert args.eval_video_root != ''
    checkpoint_path = args.pretrain_cnn_path
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    if "state_dict" in checkpoint:
        model = s3dg.S3D(
            args.num_class, space_to_depth=False, word2vec_path=args.word2vec_path)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint["state_dict"])
    else: # load pre-trained model from https://github.com/antoine77340/S3D_HowTo100M
        model = s3dg.S3D(
            args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path)
        model = torch.nn.DataParallel(model)
        checkpoint_module = {'module.' + k:v for k,v in checkpoint.items()}
        model.load_state_dict(checkpoint_module)
    model.eval()
    model.cuda()
   
    # Data loading code
    dataset = MSRVTT_DataLoader(
        data=os.path.join(os.path.dirname(__file__), 'csv/msrvtt_test.csv'),
        num_clip=args.num_windows_test,
        video_root=args.eval_video_root,
        fps=args.fps,
        num_frames=args.num_frames,
        size=args.video_size,
        crop_only=False,
        center_crop=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader,
    )
    # train for one epoch
    evaluate(dataloader, model, args)

def evaluate(train_loader, model, args):
    all_txt_embd = []
    all_video_embd = []
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(train_loader)):
            text = data['text'].cuda()
            video = data['video'].float().cuda()
            video = video / 255.0
            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4], video.shape[5])
            video_embd, text_embd = model(video, text)
            text_embd  = text_embd.cpu().numpy()
            video_embd = video_embd.view(text_embd.shape[0], args.num_windows_test, text_embd.shape[1])
            video_embd = video_embd.mean(dim=1)
            video_embd  = video_embd.cpu().numpy()
            all_txt_embd.append(text_embd)
            all_video_embd.append(video_embd)
    all_txt_embd = np.concatenate(all_txt_embd, axis=0)
    all_video_embd = np.concatenate(all_video_embd, axis=0)
    metrics = compute_metrics(np.dot(all_txt_embd, all_video_embd.T))
    print_computed_metrics(metrics)

if __name__ == "__main__":
    main()
