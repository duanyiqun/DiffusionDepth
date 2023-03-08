DiffusionDepth: Diffusion Denoising Aproach for Monocular Depth Estimation
----------


### Results

<img src="images/052.gif" width = "" height = "100" alt="图片名称" align=center /> <img src="images/036.gif" width = "" height = "100" alt="图片名称" align=center />


```shell
0022 |  Metric   |  RMSE: 1.0787  MAE: 0.4806  iRMSE: 0.0026  iMAE: 0.0017  REL: 0.0227  D^1: 0.9982  D^2: 0.9996  D^3: 0.9999
```

### Citation


```
To be revealed after review process
```


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

We used NVIDIA Apex (commit @ 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a) for multi-GPU training.

Apex can be installed as follows:

```bash
$ cd PATH_TO_INSTALL
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ git reset --hard 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 
```


#### Deformable Convolution V2 (DCNv2)

Build and install DCN module.

```bash
$ cd DiffusionDepth/src/model/deformconv
$ sh make.sh
```

The DCN module in this repository is from [here](https://github.com/xvjiarui/Deformable-Convolution-V2-PyTorch) but some function names are slightly different.

### Usage


#### Dataset

We used two datasets for training and evaluation.

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


#### KITTI Depth Prediction (KITTI DP)

KITTI DP dataset is available at the [KITTI Website](http://www.cvlibs.net/datasets/kitti).

For color images, KITTI Raw dataset is also needed, which is available at the [KITTI Raw Website](http://www.cvlibs.net/datasets/kitti/raw_data.php).
Please follow the official instructions (cf., devkit/readme.txt in each dataset) for preparation.
After downloading datasets, you should first copy color images, poses, and calibrations from the KITTI Raw to the KITTI DC dataset.

```bash
$ cd DiffusionDepth/utils
$ python prepare_KITTI_DP.py --path_root_dp PATH_TO_Dataset --path_root_raw PATH_TO_KITTI_RAW
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

```bash
$ cd DiffusionDepth/src

# An example command for NYUv2 dataset training
$ python main.py --dir_data PATH_TO_NYUv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --patch_height 228 --patch_width 304 --gpus 0,1,2,3 --loss 1.0*L1+1.0*L2 --epochs 20 \
    --batch_size 12 --max_depth 10.0 --save NAME_TO_SAVE \
    --model_name Diffusion_DCbase_ --backbone_module swin --backbone_name swin_large_naive_l4w722422k --head_specify DDIMDepthEstimate_Swin_Bins_ADDHAHI 
# An example command for KITTI DC dataset training
$ ppython main.py --dir_data datta_path --data_name KITTIDC --split_json ../data_json/kitti_dp.json \
     --patch_height 352 --patch_width 1216 --gpus 0,1,2,3 --loss 1.0*L1+1.0*L2+1.0*DDIM --epochs 28 \
     --batch_size 8 --max_depth 88.0 --save NAME_TO_SAVE \
     --model_name Diffusion_DCbase_ --backbone_module swin --backbone_name swin_large_naive_l4w722422k --head_specify DDIMDepthEstimate_Swin_Bins_ADDHAHI 
```

Please refer to the config.py for more options. Including the control of the denoising steps. 

During the training, tensorboard logs are saved under the experiments directory. To run the tensorboard:

```bash
$ cd DiffusionDepth/experiments
$ tensorboard --logdir=. --bind_all
```

#### Testing

```bash
$ cd DiffusionDepth/src

# An example command for KITTI DC dataset testing
$ python main.py --dir_data PATH_TO_KITTI_DC --data_name KITTIDC --split_json ../data_json/kitti_dp.json \
    --patch_height 240 --patch_width 1216 --gpus 0,1,2,3 --max_depth 90.0 --num_sample 0 \
    --test_only --pretrain PATH_TO_WEIGHTS --save NAME_TO_SAVE \
    --model_name Diffusion_DCbase_ --backbone_module swin --backbone_name swin_large_naive_l4w722422k --head_specify DDIMDepthEstimate_Swin_Bins_ADDHAHI 
```


#### Pre-trained Models and Results

To be revealed soon. 
