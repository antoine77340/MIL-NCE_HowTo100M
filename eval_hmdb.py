import os
import random
import time

import torch
import torch.utils.data

from args import get_args
from hmdb_loader import HMDB_DataLoader
import s3dg
from tqdm import tqdm
import numpy as np

from sklearn import preprocessing
from sklearn.svm import LinearSVC


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
    dataset = HMDB_DataLoader(
        data=os.path.join(os.path.dirname(__file__), 'csv/hmdb51.csv'),
        num_clip=args.num_windows_test,
        video_root=args.eval_video_root,
        num_frames=args.num_frames,
        size=args.video_size,
        crop_only=False,
        center_crop=True,
        with_flip=True,
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
    all_video_embd = []
    labels = []
    split1 = []
    split2 = []
    split3 = []
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(train_loader)):
            split1.append(data['split1'])
            split2.append(data['split2'])
            split3.append(data['split3'])
            labels.append(np.array(data['label']))
            video = data['video'].float().cuda()
            video = video / 255.0
            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4], video.shape[5])
            video_embd = model(video, None, mode='video', mixed5c=True)
            video_embd = video_embd.view(len(data['label']), -1, video_embd.shape[1])
            video_embd  = video_embd.cpu().numpy()
            all_video_embd.append(video_embd)
        split1 = torch.cat(split1).cpu().numpy()
        split2 = torch.cat(split2).cpu().numpy()
        split3 = torch.cat(split3).cpu().numpy()
    all_video_embd = np.concatenate(all_video_embd, axis=0)
    labels = np.concatenate(labels)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    for reg in [100.0]:
        c = LinearSVC(C=reg)
        for split in range(3):
            if split == 0:
                s = split1
            elif split == 1:
                s = split2
            else:
                s = split3
            X_train, X_test = all_video_embd[np.where(s == 1)[0]].reshape((-1, 1024)), all_video_embd[np.where(s == 2)[0]].reshape((-1, 1024))
            label_train, label_test = labels[np.where(s == 1)[0]].repeat(args.num_windows_test), labels[np.where(s == 2)[0]]
            print('Fitting SVM for split {} and C: {}'.format(split + 1, reg))
            c.fit(X_train, label_train)
            X_pred = c.decision_function(X_test)
            X_pred = np.reshape(X_pred, (len(label_test), args.num_windows_test, -1))
            X_pred = X_pred.sum(axis=1)
            X_pred = np.argmax(X_pred, axis=1)
            acc = np.sum(X_pred == label_test) / float(len(X_pred))  
            print("Top 1 accuracy split {} and C {} : {}".format(split + 1, reg, acc))

if __name__ == "__main__":
    main()
