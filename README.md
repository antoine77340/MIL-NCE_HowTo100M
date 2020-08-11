# MIL-NCE End-to-End HowTo100M training on GPUs with PyTorch

This repo contains an open source PyTorch distributed training code for the CVPR'20 paper: [End-to-End Learning of Visual Representations from Uncurated Instructional Videos](https://arxiv.org/abs/1912.06430) [1].
The original codebase from [1] relies on Google and DeepMind's internal tools as well as the usage of TPU v3 accelerators which makes it challenging to release as is.

Instead, this repository provides an implementation of [1] using PyTorch / ffmpeg with a reasonable number of GPUs.

The training code was run on the French public AI cluster [Jean-Zay](https://www.idris.fr/eng/) (see Acknowledgements below).
It was specifically designed to be run on a SLURM based cluster management for multi-node distributed training but can be easily modify for any other cluster management system.

This open source PyTorch implementation of the paper has several minor differences such as:
- The use of a cosine learning rate decay instead of a stepwise decay described in [1].
- There is no sharing of the batch normalization statistics across different GPUs and nodes as it is much slower to perform such operation on GPUs than TPUs.
- The use of slightly different spatio-temporal training video resolution of the input video clips.

If you only plan to reuse the pretrained S3D model from [1], instead please visit the following [repo](https://github.com/antoine77340/S3D_HowTo100M)
If you use this code, we would appreciate if you could both cite [1] and [2].

## Requirements

- Python 3
- PyTorch (>= 1.0)
- [python-ffmpeg](https://github.com/kkroening/ffmpeg-python) with ffmpeg 
- pandas
- tqdm (for evaluation only)
- scikit-learn (for linear evaluation only)
- SLURM cluster management for distributed training but can be easily modified for other cluster management system

## Getting preliminary word2vec and HowTo100M data for training

You will first need to download the word2vec matrix and dictionary and unzip the file in the same directory as the code, in the data folder.

```sh
wget https://www.rocq.inria.fr/cluster-willow/amiech/word2vec.zip
unzip word2vec.zip
```

Then you will need to download the preprocessed HowTo100M captions and unzip the csv files somewhere.

To download the csv files:
```sh
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/howto100m_captions.zip
```

Finally the preprocessed HowTo100M videos (12Tb in total) can be downloaded by filling this Google form: https://forms.gle/hztrfnFQUJWBtiki8.
We advise you save the HowTo100M videos as well as the caption files on fast access disks such as SSDs disks to significantly speedup the training.

## Training MIL-NCE on HowTo100M

The following command trains the S3D model on a single node, uses all of its GPU and checkpoints the model in the directory checkpoint/milnce, the log are written in the *log* directory.
Do not forget to replace *path_to_howto_csv* by the path to the HowTo100M csv caption files and *path_to_howto_videos* to the path where the HowTo100M videos were downloaded.

```sh
python main_distributed.py --n_display=1 \
       --multiprocessing-distributed --batch_size=256 \
       --num_thread_reader=40 --cudnn_benchmark=1 --pin_memory \
       --checkpoint_dir=milnce --num_candidates=4 --resume --lr=0.001 \
       --warmup_steps=10000 --epochs=300 --caption_root=path_to_howto_csv --video_path=path_to_howto_videos
```

You can also monitor the evaluation on the zero-shot YouCook2 retrieval task by specifying the argument --evaluate as well as *--eval_video_root=path_to_youcook2_video*

- Note 1: The batch size value set here is the total batch size for the node, so if batch size is 256 and there are 4 GPUs, the batch size for each GPU will be 64.
- Note 2: An epoch here is equivalent of processing 1238911 video-text training samples, which is the number of different videos in HowTo100M. It is not the same as the number of different training video clips as there are more than 100M clips. 
- Note 3: The training code should be distributed over multiple tasks with SLURM for distributed training.

## Linear evaluation of representation on HMDB-51 action recognition dataset 

### Download the videos

Please download the original HMDB videos at: [https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

### Evaluation

This evaluation will extract video features on HMDB using a model checkpoint path (to replace with *the_path_to_the_checkpoint*) and train a linear SVM using scikit-learn 
on the features.

To run the evaluation:

```sh
python eval_hmdb.py --batch_size=16  --num_thread_reader=20 --num_windows_test=10 \
        --eval_video_root=path_to_the_videos --pretrain_cnn_path=the_path_to_the_checkpoint
```
You will need to replace *path_to_the_videos* by the root folder containing the downloaded HMDB videos.

This table compares the results of the linear evaluation of the representation on HMDB-51 with the original implementation and this one under various number of training epoch and training batch size.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Implementation</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Total batch size</th>
<th valign="bottom">Accelerator</th>
<th valign="bottom">CPU cores</th>
<th valign="bottom">Training input size</th>
<th valign="bottom">Top-1 accuracy</th>
<!-- TABLE BODY -->
<tr><td align="left">TPU Tensorflow [1]</td>
<td align="center">2645</td>
<td align="center">8192</td>
<td align="center">64 x Cloud TPU v3 128Gb</td>
<td align="center">64 x N.A.</td>
<td align="center">32 frames at 200x200</td>
<td align="center">53.1</td>
</tr>
<tr><td align="left">TPU Tensorflow [1]</td>
<td align="center">206</td>
<td align="center">512</td>
<td align="center">4 x Cloud TPU v3 128Gb</td>
<td align="center">4 x N.A.</td>
<td align="center">16 frames at 200x200</td>
<td align="center">54.2</td>
</tr>
<tr><td align="left">This implementation</td>
<td align="center">150</td>
<td align="center">512</td>
<td align="center">2 x 4 Tesla V100 32Gb</td>
<td align="center">2 x 40</td>
<td align="center">16 frames at 224x224</td>
<td align="center">53.5</td>
</tr>
<tr><td align="left">This implementation</td>
<td align="center">300</td>
<td align="center">1024</td>
<td align="center">4 x 4 Tesla V100 32Gb</td>
<td align="center">4 x 40</td>
<td align="center">16 frames at 224x224</td>
<td align="center">54.3</td>
</tr>
</tbody></table>


## Zero-Shot evaluation retrieval on MSR-VTT and YouCook2

### Download the MSR-VTT videos
A mirror link of the MSR-VTT testing videos can be found at: [https://www.mediafire.com/folder/h14iarbs62e7p/shared](https://www.mediafire.com/folder/h14iarbs62e7p/shared)

### Download the YouCook2 videos
A mirror link of our downloaded YouCook2 testing videos can be found at: [https://www.rocq.inria.fr/cluster-willow/amiech/Youcook2_val.zip](https://www.rocq.inria.fr/cluster-willow/amiech/Youcook2_val.zip)

### Evaluation on MSR-VTT

This evaluation will run the zero-shot text-video retrieval on the MSR-VTT subset of the test set used in [1].
You will need to replace *the_path_to_the_checkpoint* by your model checkpoint path and *path_to_the_msrvtt_videos* to the root folder containing the downloaded MSR-VTT testing videos. 

```sh
python eval_msrvtt.py --batch_size=16  --num_thread_reader=20 --num_windows_test=10 \
       --eval_video_root=path_to_the_msrvtt_videos --pretrain_cnn_path=the_path_to_the_checkpoint
```

This table compares the retrieval results with the original implementation and this one under various number of training epoch and training batch size.

<table><tbody>

<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Implementation</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Total batch size</th>
<th valign="bottom">Accelerator</th>
<th valign="bottom">CPU cores</th>
<th valign="bottom">Training input size</th>
<th valign="bottom">R@1</th>
<th valign="bottom">R@5</th>
<th valign="bottom">R@10</th>
<th valign="bottom">Median Rank</th>
<!-- TABLE BODY -->
<tr><td align="left">TPU Tensorflow [1]</td>
<td align="center">2645</td>
<td align="center">8192</td>
<td align="center">64 x Cloud TPU v3 128Gb</td>
<td align="center">64 x N.A.</td>
<td align="center">32 frames at 200x200</td>
<td align="center">9.9</td>
<td align="center">24.0</td>
<td align="center">32.4</td>
<td align="center">29.5</td>
</tr>
<tr><td align="left">TPU Tensorflow [1]</td>
<td align="center">206</td>
<td align="center">512</td>
<td align="center">4 x Cloud TPU v3 128Gb</td>
<td align="center">4 x N.A.</td>
<td align="center">16 frames at 200x200</td>
<td align="center">8.6</td>
<td align="center">21.5</td>
<td align="center">28.6</td>
<td align="center">36</td>
</tr>
<tr><td align="left">This implementation</td>
<td align="center">150</td>
<td align="center">512</td>
<td align="center">2 x 4 Tesla V100 32Gb</td>
<td align="center">2 x 40</td>
<td align="center">16 frames at 224x224</td>
<td align="center">7.7</td>
<td align="center">19.0</td>
<td align="center">27.3</td>
<td align="center">37</td>
</tr>
<tr><td align="left">This implementation</td>
<td align="center">300</td>
<td align="center">1024</td>
<td align="center">4 x 4 Tesla V100 32Gb</td>
<td align="center">4 x 40</td>
<td align="center">16 frames at 224x224</td>
<td align="center">8.2</td>
<td align="center">21.5</td>
<td align="center">29.5</td>
<td align="center">40</td>
</tr>
</tbody></table>

### Evaluation on YouCook2

This evaluation will run the zero-shot text-video retrieval on the validation YouCook2 videos.
Please replace *the_path_to_the_checkpoint* by your model checkpoint path and *path_to_the_youcook_videos* to the root folder containing the downloaded MSR-VTT testing videos. 

```sh
python eval_youcook.py --batch_size=16  --num_thread_reader=20 --num_windows_test=10 \
        --eval_video_root=path_to_the_youcook_videos --pretrain_cnn_path=the_path_to_the_checkpoint
```

This table compares the retrieval results with the original implementation and this one under various number of training epoch and training batch size.
Note that as opposed to MSR-VTT and HMDB-51, we were not able to download same amount of YouCook2 cooking videos and thus the evaluation between the DeepMind implementation and this PyTorch 
implementation are evaluated on a slightly different number of validation videos (The DeepMind evaluation has slightly less validation videos).


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Implementation</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Total batch size</th>
<th valign="bottom">Accelerator</th>
<th valign="bottom">CPU cores</th>
<th valign="bottom">Training input size</th>
<th valign="bottom">R@1</th>
<th valign="bottom">R@5</th>
<th valign="bottom">R@10</th>
<th valign="bottom">Median Rank</th>
<!-- TABLE BODY -->
<tr><td align="left">TPU Tensorflow [1]</td>
<td align="center">2645</td>
<td align="center">8192</td>
<td align="center">64 x Cloud TPU v3 128Gb</td>
<td align="center">64 x N.A.</td>
<td align="center">32 frames at 200x200</td>
<td align="center">15.1</td>
<td align="center">38.0</td>
<td align="center">51.2</td>
<td align="center">10</td>
</tr>
<tr><td align="left">TPU Tensorflow [1]</td>
<td align="center">206</td>
<td align="center">512</td>
<td align="center">4 x Cloud TPU v3 128Gb</td>
<td align="center">4 x N.A.</td>
<td align="center">16 frames at 200x200</td>
<td align="center">8.2</td>
<td align="center">24.6</td>
<td align="center">36.2</td>
<td align="center">23</td>
</tr>
<tr><td align="left">This implementation</td>
<td align="center">150</td>
<td align="center">512</td>
<td align="center">2 x 4 Tesla V100 32Gb</td>
<td align="center">2 x 40</td>
<td align="center">16 frames at 224x224</td>
<td align="center">7.4</td>
<td align="center">21.5</td>
<td align="center">32.2</td>
<td align="center">29</td>
</tr>
<tr><td align="left">This implementation</td>
<td align="center">300</td>
<td align="center">1024</td>
<td align="center">4 x 4 Tesla V100 32Gb</td>
<td align="center">4 x 40</td>
<td align="center">16 frames at 224x224</td>
<td align="center">8.8</td>
<td align="center">24.3</td>
<td align="center">34.6</td>
<td align="center">23</td>
</tr>
</tbody></table>

## References 

[1] A. Miech, J.-B. Alayrac, L. Smaira, I. Laptev, J. Sivic and A. Zisserman,
[End-to-End Learning of Visual Representations from Uncurated Instructional Videos](https://arxiv.org/abs/1912.06430)

Presented at CVPR 2020

[2] A. Miech, D. Zhukov, J.-B. Alayrac, M. Tapaswi, I. Laptev and J. Sivic, 
[HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips](https://arxiv.org/abs/1906.03327)

Presented at ICCV 2019

Bibtex:

```bibtex
@inproceedings{miech19howto100m,
   title={How{T}o100{M}: {L}earning a {T}ext-{V}ideo {E}mbedding by {W}atching {H}undred {M}illion {N}arrated {V}ideo {C}lips},
   author={Miech, Antoine and Zhukov, Dimitri and Alayrac, Jean-Baptiste and Tapaswi, Makarand and Laptev, Ivan and Sivic, Josef},
   booktitle={ICCV},
   year={2019},
}

@inproceedings{miech19endtoend,
   title={{E}nd-to-{E}nd {L}earning of {V}isual {R}epresentations from {U}ncurated {I}nstructional {V}ideos},
   author={Miech, Antoine and Alayrac, Jean-Baptiste and Smaira, Lucas and Laptev, Ivan and Sivic, Josef and Zisserman, Andrew},
   booktitle={CVPR},
   year={2020},
}
```

## Acknowledgements

This work was granted access to the HPC resources of IDRIS under the allocation 2020-AD011011325 made by GENCI.

The SLURM distributed training code was also mainly inspired from this great [repo](https://github.com/ShigekiKarita/pytorch-distributed-slurm-example). 
