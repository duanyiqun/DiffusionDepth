Diffusion Denoising Aproach for Monocular Depth Estimation
----------

This is a pytorch implementation for paper "DiffusionDepth: Diffusion Denoising Aproach for Monocular Depth Estimation" 


### Results

---------------------------------------------------------------------------------
<img src="images/052.gif" width = "" height = "186" alt="图片名称" align=center />

---------------------------------------------------------------------------------


<img src="images/0196.gif" width = "" height = "120" alt="图片名称" align=center /><img src="images/0539.gif" width = "" height = "120" alt="图片名称" align=center /><img src="images/1019.gif" width = "" height = "120" alt="图片名称" align=center /><img src="images/0333.gif" width = "" height = "120" alt="图片名称" align=center />

---------------------------------------------------------------------------------

Best checkpoint on KITTI depth test split offline, we provide finetune metric logs in [experiments](experiments)
```shell
0022 |  Metric   |  RMSE: 1.0787  MAE: 0.4806  iRMSE: 0.0026  iMAE: 0.0017  REL: 0.0227  D^1: 0.9982  D^2: 0.9996  D^3: 0.9999
```

### Citation


```
To be revealed soon
```

---------------------------------------------------------------------------------
### Dependencies

Our released implementation is tested on:

- Ubuntu 16.04 / Ubuntu 18.04
- Python 3.8 (Anaconda 4.8.4)
- PyTorch 1.9 / torchvision 0.7
- Tensorboard 2.3
- NVIDIA CUDA 11.0
- NVIDIA Apex
- [Deformable Convolution V2](https://arxiv.org/abs/1811.11168)
- 8x NVIDIA GTX 3090 / 8x NVIDIA A100 RTX GPUs


#### NVIDIA Apex

NVIDIA Apex is a good choice for multi-GPU training to save GPU memory. However we only use option "O0" to train the work. Wellcome to discuss with us about half precision performance. 
Apex can be installed as follows:

```bash
$ cd PATH_TO_INSTALL
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 
```

---------------------------------------------------------------------------------

### Usage


#### Dataset

We used two datasets for training and evaluation. Please also see [nlspn](https://github.com/zzangjinsun/NLSPN_ECCV20) with there excellent work on data processing.  

#### NYU Depth V2 (NYUv2)

We used preprocessed NYUv2 HDF5 dataset provided by [Fangchang Ma](https://github.com/fangchangma/sparse-to-dense).

```bash
$ cd PATH_TO_DOWNLOAD
$ wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
$ tar -xvf nyudepthv2.tar.gz
```
After preparing the dataset, you should generate a json file containing paths to individual images.

```bash
$ cd DiffusionDepth/utils
$ python generate_json_NYUDepthV2.py --path_root PATH_TO_NYUv2
```
After that, you will get a data structure as follows:

```
nyudepthv2
├── train
│    ├── basement_0001a
│    │    ├── 00001.h5
│    │    └── ...
│    ├── basement_0001b
│    │    ├── 00001.h5
│    │    └── ...
│    └── ...
└── val
    └── official
        ├── 00001.h5
        └── ...
```

#### KITTI Depth Prediction (KITTI DP)

KITTI DP dataset is available at the [KITTI Website](http://www.cvlibs.net/datasets/kitti). We should choose depth prediction for re-implementation. 

For color images, KITTI Raw dataset is also needed, which is available at the [KITTI Raw Website](http://www.cvlibs.net/datasets/kitti/raw_data.php).
Please follow the official instructions (cf., devkit/readme.txt in each dataset) for preparation.
After downloading datasets, you should first copy color images, poses, and calibrations from the KITTI Raw to the KITTI Depth dataset.

```bash
$ cd DiffusionDepth/utils
$ python prepare_KITTI_DP.py --path_root_dp PATH_TO_Dataset --path_root_raw PATH_TO_KITTI_RAW
```

```
.
├── depth_selection
│    ├── test_depth_completion_anonymous
│    │    ├── image
│    │    ├── intrinsics
│    │    └── velodyne_raw
│    ├── test_depth_prediction_anonymous
│    │    ├── image
│    │    └── intrinsics
│    └── val_selection_cropped
│        ├── groundtruth_depth
│        ├── image
│        ├── intrinsics
│        └── velodyne_raw
├── train
│    ├── 2011_09_26_drive_0001_sync
│    │    ├── image_02
│    │    │     └── data
│    │    ├── image_03
│    │    │     └── data
│    │    ├── oxts
│    │    │     └── data
│    │    └── proj_depth
│    │        ├── groundtruth
│    │        └── velodyne_raw
│    └── ...
└── val
    ├── 2011_09_26_drive_0002_sync
    └── ...
```

After preparing the dataset, you should generate a json file containing paths to individual images.

```bash
$ cd DiffusionDepth/utils

# For Train / Validation
$ python generate_json_KITTI_DP.py --path_root PATH_TO_KITTI

# For Online Evaluation Data
$ python generate_json_KITTI_DP.py --path_root PATH_TO_KITTI --name_out kitti_dp_test.json --test_data
```


#### Training

Notes, recomended to download pretrain from Swin transformer official website and modify the backbone file [src/model/backbone/swin.py](src/model/backbone/swin.py) replacing the pretrain path to your local copy.

```bash
$ cd DiffusionDepth/src
# An example command for KITTI dataset training
$ ppython main.py --dir_data datta_path --data_name KITTIDC --split_json ../data_json/kitti_dp.json \
     --patch_height 352 --patch_width 906 --gpus 0,1,2,3 --loss 1.0*L1+1.0*L2+1.0*DDIM --epochs 30 \
     --batch_size 8 --max_depth 88.0 --save NAME_TO_SAVE \
     --model_name Diffusion_DCbase_ --backbone_module swin --backbone_name swin_large_naive_l4w722422k --head_specify DDIMDepthEstimate_Swin_ADDHAHI 
```
Please refer to the config.py for more options. Including the control of the denoising steps ```--inference_steps=20``` and training diffusion steps ```--num_train_timesteps=1000```. 


During the training, tensorboard logs are saved under the experiments directory. To run the tensorboard:

```bash
$ cd DiffusionDepth/experiments
$ tensorboard --logdir=. --bind_all
```

#### Testing
With only batch size 1 is recomended. 
```bash
$ cd DiffusionDepth/src

# An example command for KITTI DC dataset testing
$ python main.py --dir_data PATH_TO_KITTI_DC --data_name KITTIDC --split_json ../data_json/kitti_dp.json \
    --patch_height 352 --patch_width 1216 --gpus 0,1,2,3 --max_depth 80.0 --num_sample 0 --batch_size 1\
    --test_only --pretrain PATH_TO_WEIGHTS --save NAME_TO_SAVE --save_image\
    --model_name Diffusion_DCbase_ --backbone_module swin --backbone_name swin_large_naive_l4w722422k --head_specify DDIMDepthEstimate_Swin_ADDHAHI 
```


#### Pre-trained Models and Results

To be revealed soon. 


#### Notes
Thanks [nlspn](https://github.com/zzangjinsun/NLSPN_ECCV20) with there excellent work. This code base borrow borrow frameworks to accelerate implementation. 