# inverse σSRN

This is the official code repository for the paper "Generalizable Pose Estimation Using Implicit Scene Representations" presented at ICRA 2023.

<img src="https://github.com/vaibhavsaxena11/inverse-sigma-srn/assets/10887480/6ded7ac0-0829-484f-94f8-b0e281d0669a" height="200">
<img src="https://github.com/vaibhavsaxena11/inverse-sigma-srn/assets/10887480/d383d48e-4ff1-4ae6-ae55-ae61b06da008" height="200">

Please see our [project website](https://sites.google.com/view/generalizable-pose-estimation) for demo videos and more details about our method! 

## Usage
The code for the σSRN model is based on the [original SRN codebase](https://github.com/vsitzmann/scene-representation-networks). The modified code is located in the sigma_srn folder. 

### Installation
```
git clone --recurse-submodules https://github.com/vaibhavsaxena11/inverse-sigma-srn.git
```

```
conda env create -f environment.yml
```

```
conda activate sigma-srns
```

### Data
Download Shapenet v2 cars and chairs classes dataset from [here](https://drive.google.com/drive/folders/1OkYgeRcIcLOFu1ft5mRODWNQaPJ0ps90?usp=sharing). Fetch the cars_train.zip, cars_train_val.zip, chairs_train.zip and chairs_train_val.zip files.

### Training a σSRN Model
See `python train.py --help` for all train options. 
Example train call:
```
python train.py --data_root [path to directory with dataset] \
                --logging_root [path to directory where tensorboard summaries and checkpoints should be written to] \
                --no_validation \
                --img_sidelengths 64 \
                --batch_size_per_img_sidelength 10 
```
To monitor progress, the training code writes tensorboard summaries every 100 steps into a "events" subdirectory in the logging_root.

### Pose Estimation through inverse σSRN
See `python pose.py --help` for all pose evaluation options. 
Example pose call:
```
python pose.py --train_data_root [path to directory with dataset] \
               --val_data_root [path to directory with train_val dataset - unseen poses for evaluation] \
               --logging_root [path to directory where the results are stored] \
               --checkpoint_path [path to the σSRN model checkpoint *.pth file] 
               --loss l1
```

For each image in the validation set (from an unseen camera pose), pose.py will optimize the camera pose parameters to move the view to match the input image. Multiple starting poses are used, and the pose with the lowest loss is chosen as the estimated camera pose associated with the input image. 
For each pose optimization, the rendered views as the camera parameters move towards the target pose are stored in the logging directory. If plotting is enabled (on by default), error plots are stored in the logging directory as well. For each input target image, a pose.txt file indicates the best estimate.

## Citation

If you find this repository useful, please cite our work:

```bibtex
@misc{saxena2023generalizable,
      title={Generalizable Pose Estimation Using Implicit Scene Representations}, 
      author={Vaibhav Saxena and Kamal Rahimi Malekshan and Linh Tran and Yotto Koga},
      year={2023},
      eprint={2305.17252},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
