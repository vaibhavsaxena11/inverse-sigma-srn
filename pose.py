"""
Utility to estimate camera pose using a trained sigma-SRN on unseen poses.
"""
from time import time
import os
from torch.utils.data import DataLoader
import configargparse
import torch
from functorch import make_functional_with_buffers
import numpy as np
from invert_renderer import invert

from sigma_srn import dataio
from sigma_srn.srns import SRNsModel
from sigma_srn import util

p = configargparse.ArgumentParser()
p.add_argument(
    "--train_data_root", required=True, help="Path to directory with training data."
)
p.add_argument(
    "--val_data_root", required=True, help="Path to directory with validation data."
)
p.add_argument(
    "--logging_root",
    required=True,
    help="Path to directory where optimization"
    "progress and pose estimates will be saved.",
)
p.add_argument("--checkpoint_path", required=True, help="Path to trained model.")
p.add_argument("--gpu_id", type=int, default=0, help="Gpu id.")
p.add_argument(
    "--lr", type=float, default=1e-2, help="Learning rate for the pose optimization."
)
p.add_argument(
    "--num_steps", type=int, default=300, help="Number of steps in the optimization."
)
p.add_argument("--plot", default=True, help="Plot the error curve.")
p.add_argument(
    "--loss", default="mse", help="Loss: mse, l1, norm_l1, norm_l2, gmsd_loss."
)
p.add_argument(
    "--four_neigh_offset",
    type=float,
    default=0.524,
    help="Rotation offset from the target pose."
    "Takes effect when use_24_start_poses is not enabled.",
)
p.add_argument(
    "--use_24_start_poses",
    action="store_true",
    default=False,
    help="Use 24 start poses for each pose estimate.",
)
# shapenet cars radius = 1.3, shapenet chairs radius = 2.0
p.add_argument(
    "--radius",
    type=float,
    default=1.3,
    help="Camera orbit radius in the datatset."
    "This should be specified when use_24_start_poses is enabled.",
)

opt = p.parse_args()

# fixed settings
DISABLE_SIGMA = False
MAX_NUM_INSTANCES_TRAIN = -1
DISABLE_MAKEGRID = True
MAX_NUM_OBSERVATIONS_TRAIN = 50
# evaluation protocol is 10 instances x 10 unseen poses
MAX_NUM_INSTANCES_VAL = 10
MAX_NUM_OBSERVATIONS_VAL = 10
IMG_SIDELENGTHS = 64
BATCH_SIZE = 5
SPECIFIC_OBSERVATION_IDCS = None
EMBEDDING_SIZE = 256
FIT_SINGLE_SRN = False
USE_UNET_RENDERER = False
TRACING_STEPS = 10
FREEZE_NETWORKS = True
OVERWRITE_EMBEDDINGS = False

torch.cuda.set_device(opt.gpu_id)

# Setup data loaders
NUM_OUTPUT_CHANNELS = 3
train_dataset = dataio.SceneClassDataset(
    root_dir=opt.train_data_root,
    max_num_instances=MAX_NUM_INSTANCES_TRAIN,
    max_observations_per_instance=MAX_NUM_OBSERVATIONS_TRAIN,
    img_sidelength=IMG_SIDELENGTHS,
    specific_observation_idcs=SPECIFIC_OBSERVATION_IDCS,
    samples_per_instance=1,
    num_output_channels=NUM_OUTPUT_CHANNELS,
)

# test set of unseen poses (but seen objects)
val_dataset = dataio.SceneClassDataset(
    root_dir=opt.val_data_root,
    max_num_instances=MAX_NUM_INSTANCES_VAL,
    max_observations_per_instance=MAX_NUM_OBSERVATIONS_VAL,
    img_sidelength=IMG_SIDELENGTHS,
    specific_observation_idcs=SPECIFIC_OBSERVATION_IDCS,
    samples_per_instance=1,
    num_output_channels=NUM_OUTPUT_CHANNELS,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    collate_fn=val_dataset.collate_fn,
    pin_memory=False,
)
val_iterator = iter(val_dataloader)

# Load Model
with torch.cuda.device(opt.gpu_id):
    model = SRNsModel(
        num_instances=train_dataset.num_instances,
        latent_dim=EMBEDDING_SIZE,
        has_params=False,
        fit_single_srn=FIT_SINGLE_SRN,
        use_unet_renderer=USE_UNET_RENDERER,
        tracing_steps=TRACING_STEPS,
        freeze_networks=FREEZE_NETWORKS,  # set to True when just using the model for inference
        disable_sigma=DISABLE_SIGMA,
        disable_makegrid=DISABLE_MAKEGRID,
        num_output_channels=NUM_OUTPUT_CHANNELS,
    )
model.cuda()

if opt.checkpoint_path is not None:
    print(f"Loading model from {opt.checkpoint_path}")
    util.custom_load(
        model,
        path=opt.checkpoint_path,
        discriminator=None,
        optimizer=None,
        overwrite_embeddings=OVERWRITE_EMBEDDINGS,
    )

# Creating functional model
fmodel, params, buffers = make_functional_with_buffers(model)
model = [fmodel, params, buffers]

base_dir = opt.logging_root
all_rot_errors = []
all_tra_errors = []
for i, (model_input, _) in enumerate(val_iterator):
    print(f"Example {i}\n")
    t_start = time()
    cam_pose, rot_error, tra_error, init_idx = invert(
        model, model_input, opt, BATCH_SIZE * i, NUM_OUTPUT_CHANNELS
    )
    t_end = time()
    all_rot_errors.append(rot_error)
    all_tra_errors.append(tra_error)

avg_rot_error = np.mean(all_rot_errors)
avg_tra_error = np.mean(all_tra_errors)
with open(os.path.join(base_dir, "errors.txt"), "w", encoding="utf-8") as f:
    f.write(f"num examples = {len(val_iterator)*BATCH_SIZE}\n")
    f.write(f"average rotation error = {avg_rot_error}\n")
    f.write(f"average translation error = {avg_tra_error}\n")
    f.write(f"time elapsed (last batch) = {t_end-t_start}")

print(f"num examples = {len(val_iterator)*BATCH_SIZE}")
print(f"average rotation error = {avg_rot_error}")
print(f"average translation error = {avg_tra_error}")
print(f"time elapsed (last batch) = {t_end-t_start}")
print(f"loss = {opt.loss}")
print(f"num steps = {opt.num_steps}")
