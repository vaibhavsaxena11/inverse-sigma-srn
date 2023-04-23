"""
Utility to update camera pose parameters (Euler rx,ry,rz and translation px,py,pz) to
match a target input image with rendered sigma-SRN. Pose estimation.
"""
import logging
import os
from time import time
import numpy as np
import torch
from functorch import vmap, grad_and_value
import matplotlib.pyplot as plt
from piq import GMSDLoss
import cam_util as cu

logger = logging.getLogger()
old_level = logger.level


def rot_error(r_est, r_gt):
    """
    Courtesy of PoseCNN.
    Rotational Error.
    :param R_est: Rotational element of the estimated pose (3x1 vector).
    :param R_gt: Rotational element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert r_est.shape == r_gt.shape == (3, 3)
    # Explanation: in 3D, trace of a rotation matrix = 1 + 2*cos(theta),
    # where theta is the rotation angle
    error_cos = 0.5 * (torch.trace(torch.matmul(r_est, torch.linalg.inv(r_gt))) - 1.0)
    error_cos = torch.clamp(error_cos, -1, 1)
    error = torch.acos(error_cos)
    error = 180.0 * error / np.pi  # [rad] -> [deg]
    return error


batched_re = vmap(rot_error)


def translation_error(t_est, t_gt):
    """
    Courtesy of PoseCNN.
    Translational Error.
    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert t_est.shape == t_gt.shape == (3,)
    error = torch.linalg.norm(t_gt - t_est)
    return error


batched_te = vmap(translation_error)


def _get_ft_pose_gradients(model, input_val, target, loss_fn, cam_pose):
    fmodel, params, buffers = model

    # we don't need to store grads for backprop
    # avoiding this step might result in a memory leak
    for param in params:
        param.requires_grad_(False)

    def compute_loss_stateless_model(params, buffers, sample, cam_pose, target):
        # compute the camera matrix from the 6 dof camera pose parameters
        # first 3 terms are the euler angles representing the rotation
        # the last 3 terms are the x,y,z translation values
        c1 = torch.cos(cam_pose[:, 0])
        c2 = torch.cos(cam_pose[:, 1])
        c3 = torch.cos(cam_pose[:, 2])
        s1 = torch.sin(cam_pose[:, 0])
        s2 = torch.sin(cam_pose[:, 1])
        s3 = torch.sin(cam_pose[:, 2])
        tx = cam_pose[:, 3]
        ty = cam_pose[:, 4]
        tz = cam_pose[:, 5]
        zero = torch.tensor([0])
        one = torch.tensor([1])
        cam_mat = torch.cat(
            (
                c2 * c3,
                -c2 * s3,
                s2,
                tx,
                c1 * s3 + c3 * s1 * s2,
                c1 * c3 - s1 * s2 * s3,
                -c2 * s1,
                ty,
                s1 * s3 - c1 * c3 * s2,
                c3 * s1 + c1 * s2 * s3,
                c1 * c2,
                tz,
                zero,
                zero,
                zero,
                one,
            )
        ).reshape((4, 4))
        sample["pose"] = cam_mat

        batch = dict([(k, v.unsqueeze(0)) for k, v in sample.items()])
        targets = target.unsqueeze(0)

        preds_rgb, _ = fmodel(params, buffers, batch)
        loss = loss_fn(preds_rgb, targets)
        return loss

    ft_compute_grad = grad_and_value(compute_loss_stateless_model, argnums=3)
    ft_compute_sample_grad = vmap(
        ft_compute_grad,
        in_dims=(None, None, dict.fromkeys(input_val.keys(), 0), 0, 0),
        randomness="same",
    )
    ft_per_sample_grads, ft_per_sample_loss = ft_compute_sample_grad(
        params, buffers, input_val, cam_pose, target
    )

    return ft_per_sample_loss, ft_per_sample_grads


def _get_ft_srns_output(model, input_val):
    fmodel, params, buffers = model
    return fmodel(params, buffers, input_val)[0]


gmsd = GMSDLoss()


def gmsd_loss(output, target):
    """
    Gradient Magnitude Similarity Deviation loss
    """
    x = (output.reshape(1, 64, 64, 3) + 1) / 2
    y = (target.reshape(1, 64, 64, 3) + 1) / 2
    return gmsd(x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2))


def normalized_l1_loss(output, target):
    """
    Normalized L1 loss (in the spirit of focal loss)
    """
    loss = torch.sum(torch.abs(output - target)) / (torch.count_nonzero(target - 1) + 1)
    return loss


def normalized_l2_loss(output, target):
    """
    Normalized L2 loss (in the spirit of focal loss)
    """
    loss = torch.sum((output - target) ** 2) / (torch.count_nonzero(target - 1) + 1)
    return loss


def gmsd_normalized_l1_loss(output, target):
    """
    A blend of gmsd and normalized L1 loss
    """
    alpha = 0.75
    return alpha * gmsd_loss(output, target) + (1.0 - alpha) * normalized_l1_loss(
        output, target
    )


def invert(model, input_val, model_config, batch_start_idx, num_output_channels):
    # Setup logging
    logdir = model_config.logging_root
    os.makedirs(logdir, exist_ok=True)

    image_shape = (64, 64)
    assert image_shape[0] * image_shape[1] == input_val["rgb"].shape[-2], input_val[
        "rgb"
    ].shape

    target_pose = input_val["pose"].detach().cpu()
    batch_size, _ = input_val["rgb"].shape[:2]
    if not model_config.use_24_start_poses:
        # start pose for the camera is one of four poses 40 degrees from the target
        # NB: we just need len(init_poses) == 4 to iterate through the poses
        init_poses = [1, 1, 1, 1]
    else:
        # a collection of 24 6 dof poses for the camera looking at the object
        # from different locations on a unit sphere (object is centered at the origin)
        init_poses = [
            torch.tensor(
                [-3.141593, -0.785398, -1.570796, 0.707107, 0.000000, 0.707107]
            ),
            torch.tensor([2.356194, 0.000000, -3.141593, 0.000000, 0.707107, 0.707107]),
            torch.tensor(
                [-3.141593, 0.785398, 1.570796, -0.707107, 0.000000, 0.707107]
            ),
            torch.tensor(
                [-2.356194, 0.000000, -0.000000, 0.000000, -0.707107, 0.707107]
            ),
            torch.tensor(
                [-0.000000, -0.785398, 1.570796, 0.707107, 0.000000, -0.707107]
            ),
            torch.tensor(
                [0.785398, 0.000000, -3.141593, 0.000000, 0.707107, -0.707107]
            ),
            torch.tensor(
                [-0.000000, 0.785398, -1.570796, -0.707107, 0.000000, -0.707107]
            ),
            torch.tensor(
                [-0.785398, 0.000000, -0.000000, 0.000000, -0.707107, -0.707107]
            ),
            torch.tensor(
                [2.526113, -0.523599, -2.526113, 0.500000, 0.500000, 0.707107]
            ),
            torch.tensor([2.526113, 0.523599, 2.526113, -0.500000, 0.500000, 0.707107]),
            torch.tensor(
                [-2.526113, 0.523599, 0.615480, -0.500000, -0.500000, 0.707107]
            ),
            torch.tensor(
                [-2.526113, -0.523599, -0.615480, 0.500000, -0.500000, 0.707107]
            ),
            torch.tensor(
                [0.615480, -0.523599, 2.526113, 0.500000, 0.500000, -0.707107]
            ),
            torch.tensor(
                [0.615480, 0.523599, -2.526113, -0.500000, 0.500000, -0.707107]
            ),
            torch.tensor(
                [-0.615480, 0.523599, -0.615480, -0.500000, -0.500000, -0.707107]
            ),
            torch.tensor(
                [-0.615480, -0.523599, 0.615480, 0.500000, -0.500000, -0.707107]
            ),
            torch.tensor(
                [-1.570796, -1.570796, 0.000000, 1.000000, 0.000000, 0.000000]
            ),
            torch.tensor([1.570796, 0.000000, -3.141593, 0.000000, 1.000000, 0.000000]),
            torch.tensor(
                [-1.570796, 1.570796, 0.000000, -1.000000, 0.000000, 0.000000]
            ),
            torch.tensor(
                [-1.570796, 0.000000, -0.000000, 0.000000, -1.000000, 0.000000]
            ),
            torch.tensor(
                [1.570796, -0.785398, -3.141593, 0.707107, 0.707107, 0.000000]
            ),
            torch.tensor(
                [1.570796, 0.785398, -3.141593, -0.707107, 0.707107, 0.000000]
            ),
            torch.tensor(
                [-1.570796, 0.785398, -0.000000, -0.707107, -0.707107, 0.000000]
            ),
            torch.tensor(
                [-1.570796, -0.785398, -0.000000, 0.707107, -0.707107, 0.000000]
            ),
        ]

    all_init_camposes = []
    all_init_start_loss = []
    all_init_end_loss = []
    all_init_start_rot_errors = []
    all_init_end_rot_errors = []
    all_init_start_tra_errors = []
    all_init_end_tra_errors = []
    all_init_end_ious = []

    # create the eval directories and store the target (from the validation dataset and
    # the corresponding rendered target - should be the same or as good as SRN can render)
    rgb = _get_ft_srns_output(model, input_val)
    for i_init in range(len(init_poses)):
        for i_batch in range(batch_size):
            os.makedirs(
                os.path.join(logdir, str(i_batch + batch_start_idx), str(i_init)),
                exist_ok=True,
            )
            sublogdir = os.path.join(
                logdir, str(i_batch + batch_start_idx), str(i_init)
            )

            plt.figure()
            plt.title("Input RGB")
            logger.setLevel(100)
            if num_output_channels == 3:
                plt.imshow(
                    input_val["rgb"][i_batch]
                    .reshape(image_shape + (num_output_channels,))
                    .detach()
                    .cpu()
                    .numpy()
                    * 0.5
                    + 0.5
                )
            else:
                plt.imshow(
                    input_val["rgb"][i_batch]
                    .reshape(image_shape)
                    .detach()
                    .cpu()
                    .numpy()
                    * 0.5
                    + 0.5,
                    cmap="gray",
                )
            logger.setLevel(old_level)
            plt.savefig(os.path.join(sublogdir, "input.png"))

            # input["pose"] has the target pose from the loader
            plt.figure()
            plt.title("RGB rendering at target pose")
            logger.setLevel(100)
            if num_output_channels == 3:
                plt.imshow(
                    rgb[i_batch]
                    .reshape(image_shape + (num_output_channels,))
                    .detach()
                    .cpu()
                    .numpy()
                    * 0.5
                    + 0.5
                )
            else:
                plt.imshow(
                    rgb[i_batch].reshape(image_shape).detach().cpu().numpy() * 0.5
                    + 0.5,
                    cmap="gray",
                )
            logger.setLevel(old_level)
            plt.savefig(os.path.join(sublogdir, "rgb_targetpose.png"))

    for i_init, input_pose in enumerate(init_poses):
        if not model_config.use_24_start_poses:
            poses = []
            for b in range(batch_size):
                target = target_pose[b].reshape(16)
                world_T_target = cu.Transform.from_matrix(target.tolist())
                # from the target pose, choose a start point for the optimization,
                # offset by four_neigh_offset radians in the down, up, right and left directions
                (
                    rot,
                    pos,
                ) = cu.shift_world_T_camera(
                    world_T_target, model_config.four_neigh_offset, i_init
                )
                start_pose = torch.tensor(
                    [rot[2], rot[1], rot[0], pos[0], pos[1], pos[2]], dtype=torch.float
                ).reshape(1, 6)
                poses.append(start_pose)

            cam_pose = torch.stack(poses, dim=0)
        else:
            # the camera init_poses are defined at a distance of 1 from the object center
            # for cars and chairs the camera is actually at a different distance so apply
            # that distance to the init pose to be consistent
            input_pose[3] *= model_config.radius
            input_pose[4] *= model_config.radius
            input_pose[5] *= model_config.radius
            cam_pose = torch.clone(input_pose.detach()).unsqueeze(0)
            cam_pose = torch.tile(cam_pose, (batch_size, 1, 1))  # .cuda()
        cam_pose.requires_grad = False

        # Create loss and optimizer for pose.
        optimizer = torch.optim.Adam(params=[cam_pose], lr=model_config.lr)

        # default is mse loss, but the normalized L1 loss works good too
        if model_config.loss == "l1":
            plot_loss_title = "L1"
            loss_fn = torch.nn.L1Loss()
        elif model_config.loss == "norm_l1":
            plot_loss_title = "Normalized L1"
            loss_fn = normalized_l1_loss
        elif model_config.loss == "norm_l2":
            plot_loss_title = "Normalized L2"
            loss_fn = normalized_l2_loss
        elif model_config.loss == "gmsd_loss":
            plot_loss_title = "Gradient Magnitude Similarity Deviation"
            loss_fn = gmsd_loss
        else:
            plot_loss_title = "Mean Squared Error"
            loss_fn = torch.nn.MSELoss()

        # Flatten target image for loss computation.
        target_image_flatten = torch.reshape(
            input_val["rgb"], [-1, input_val["rgb"].shape[-2], num_output_channels]
        )
        target_image_flatten = target_image_flatten.cuda()
        losses = []
        rot_errors = []
        tra_errors = []

        t_total = 0.0
        for step in range(model_config.num_steps):
            t_start = time()
            loss, pose_grads = _get_ft_pose_gradients(
                model, input_val, target_image_flatten, loss_fn, cam_pose
            )
            cam_pose.grad = pose_grads

            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
            cam_pose.grad = None

            # update the camera matrices from the updated pose
            c1 = torch.cos(cam_pose[:, :, 0])
            c2 = torch.cos(cam_pose[:, :, 1])
            c3 = torch.cos(cam_pose[:, :, 2])
            s1 = torch.sin(cam_pose[:, :, 0])
            s2 = torch.sin(cam_pose[:, :, 1])
            s3 = torch.sin(cam_pose[:, :, 2])
            tx = cam_pose[:, :, 3]
            ty = cam_pose[:, :, 4]
            tz = cam_pose[:, :, 5]
            zero = torch.zeros(5, 1)
            one = torch.ones(5, 1)
            cam_mat = torch.cat(
                (
                    c2 * c3,
                    -c2 * s3,
                    s2,
                    tx,
                    c1 * s3 + c3 * s1 * s2,
                    c1 * c3 - s1 * s2 * s3,
                    -c2 * s1,
                    ty,
                    s1 * s3 - c1 * c3 * s2,
                    c3 * s1 + c1 * s2 * s3,
                    c1 * c2,
                    tz,
                    zero,
                    zero,
                    zero,
                    one,
                ),
                dim=1,
            ).reshape(5, 4, 4)
            input_val["pose"] = cam_mat
            t_total += time() - t_start

            rot_errors.append(
                batched_re(cam_mat[:, :3, :3].detach().cpu(), target_pose[:, :3, :3])
            )
            tra_errors.append(
                batched_te(cam_mat[:, :3, 3].detach().cpu(), target_pose[:, :3, 3])
            )

            if step % 10 == 0 or step == model_config.num_steps - 1:
                rgb = _get_ft_srns_output(model, input_val)
                print(
                    f"Step {step}, avg image loss: {loss}, grad norm:"
                    f"{pose_grads.norm(2).item():.10f}"
                )
                for i_batch in range(batch_size):
                    sublogdir = os.path.join(
                        logdir, str(i_batch + batch_start_idx), str(i_init)
                    )
                    plt.figure()
                    plt.title(f"Step {step}")
                    logger.setLevel(100)
                    if num_output_channels == 3:
                        plt.imshow(
                            rgb[i_batch]
                            .reshape(image_shape + (num_output_channels,))
                            .detach()
                            .cpu()
                            .numpy()
                            * 0.5
                            + 0.5
                        )
                    else:
                        plt.imshow(
                            rgb[i_batch].reshape(image_shape).detach().cpu().numpy()
                            * 0.5
                            + 0.5,
                            cmap="gray",
                        )
                    logger.setLevel(old_level)
                    plt.savefig(os.path.join(sublogdir, f"rgb_{step}.png"))

            if step == model_config.num_steps - 1:
                assert rgb.shape[-1] == num_output_channels, rgb.shape
                mask_rgb = torch.sum(rgb, -1) > 0
                mask_tgt = torch.sum(target_image_flatten, -1) > 0
                union = torch.sum(torch.logical_or(mask_rgb, mask_tgt), -1)
                inter = torch.sum(torch.logical_and(mask_rgb, mask_tgt), -1)
                all_init_end_ious.append(
                    inter.detach().cpu().numpy() / union.detach().cpu().numpy()
                )
        print(
            f"Time taken per step of pose inference:\
               {1000*t_total/model_config.num_steps:.2f}ms"
        )

        ## END of testing one init for the entire batch; now plotting --
        if model_config.plot:
            for i_batch in range(batch_size):
                sublogdir = os.path.join(
                    logdir, str(i_batch + batch_start_idx), str(i_init)
                )

                plt.figure()
                plt.plot(
                    np.arange(0, model_config.num_steps, 1),
                    [losses[i_step][i_batch] for i_step in range(len(losses))],
                )

                plt.title(plot_loss_title)
                plt.savefig(os.path.join(sublogdir, "plot_loss.png"))

                plt.figure()
                plt.plot(
                    np.arange(0, model_config.num_steps, 1),
                    [rot_errors[i_step][i_batch] for i_step in range(len(rot_errors))],
                )
                plt.title("Rotation Error")
                plt.xlabel("Step")
                plt.ylabel("Error (degrees)")
                plt.savefig(os.path.join(sublogdir, "plot_rot.png"))

                plt.figure()
                plt.plot(
                    np.arange(0, model_config.num_steps, 1),
                    [tra_errors[i_step][i_batch] for i_step in range(len(tra_errors))],
                )
                plt.title("Translation Error")
                plt.xlabel("Step")
                plt.ylabel("Error")
                plt.savefig(os.path.join(sublogdir, "plot_tra.png"))

                # store the loss, rot and translation errors to a file
                with open(
                    os.path.join(sublogdir, "losses.txt"), "w", encoding="utf-8"
                ) as output_file:
                    output_file.write(
                        "\n".join(
                            str(losses[i_step][i_batch])
                            for i_step in range(len(losses))
                        )
                    )
                with open(
                    os.path.join(sublogdir, "rot_errors.txt"), "w", encoding="utf-8"
                ) as output_file:
                    output_file.write(
                        "\n".join(
                            str(rot_errors[i_step][i_batch].item())
                            for i_step in range(len(rot_errors))
                        )
                    )
                with open(
                    os.path.join(sublogdir, "tra_errors.txt"), "w", encoding="utf-8"
                ) as output_file:
                    output_file.write(
                        "\n".join(
                            str(tra_errors[i_step][i_batch].item())
                            for i_step in range(len(tra_errors))
                        )
                    )

        all_init_camposes.append(cam_pose.detach().cpu().numpy())
        all_init_start_loss.append(losses[0])
        all_init_end_loss.append(losses[-1])
        all_init_start_rot_errors.append(rot_errors[0].detach().cpu().numpy())
        all_init_end_rot_errors.append(rot_errors[-1].detach().cpu().numpy())
        all_init_start_tra_errors.append(tra_errors[0].detach().cpu().numpy())
        all_init_end_tra_errors.append(tra_errors[-1].detach().cpu().numpy())

        # best solution is the one with the lowest loss
        chosen_inits = np.argmin(all_init_end_loss, 0)
        best_rot_inits = np.argmin(all_init_end_rot_errors, 0)
        best_pos_inits = np.argmin(all_init_end_tra_errors, 0)

        write_to_files(
            target_pose,
            all_init_camposes,
            all_init_start_loss,
            all_init_end_loss,
            all_init_start_rot_errors,
            all_init_end_rot_errors,
            all_init_start_tra_errors,
            all_init_end_tra_errors,
            all_init_end_ious,
            chosen_inits,
            best_rot_inits,
            best_pos_inits,
            batch_start_idx,
            logdir,
        )
    return (
        [all_init_camposes[c][b] for c, b in zip(chosen_inits, range(batch_size))],
        [
            all_init_end_rot_errors[c][b]
            for c, b in zip(chosen_inits, range(batch_size))
        ],
        [
            all_init_end_tra_errors[c][b]
            for c, b in zip(chosen_inits, range(batch_size))
        ],
        chosen_inits,
    )


def write_to_files(
    target_poses,
    all_init_camposes,
    all_init_start_loss,
    all_init_end_loss,
    all_init_start_rot_errors,
    all_init_end_rot_errors,
    all_init_start_tra_errors,
    all_init_end_tra_errors,
    all_init_end_ious,
    chosen_inits,
    best_rot_inits,
    best_pos_inits,
    batch_start_idx,
    base_dir,
):
    """
    Utility to save evaluation results for each pose optimization, and the
    best (based on lowest loss) result from the set of initial starting camera poses
    for the pose estimation task.
    """
    num_inits = len(all_init_camposes)
    batch_size = all_init_camposes[0].shape[0]

    for i_batch in range(batch_size):
        for i_init in range(num_inits):
            sublogdir = os.path.join(
                base_dir, str(i_batch + batch_start_idx), str(i_init)
            )
            with open(os.path.join(sublogdir, "pose.txt"), "w", encoding="utf-8") as f:
                f.write(f"target pose = {target_poses[i_batch]}\n")
                f.write(f"predicted pose = {all_init_camposes[i_init][i_batch]}\n")
                f.write(f"initial loss = {all_init_start_loss[i_init][i_batch]}\n")
                f.write(
                    f"initial rotation error = {all_init_start_rot_errors[i_init][i_batch]}\n"
                )
                f.write(
                    f"initial translation error =\
                         {all_init_start_tra_errors[i_init][i_batch]}\n"
                )
                f.write(f"final iou = {all_init_end_ious[i_init][i_batch]}\n")
                f.write(f"final loss = {all_init_end_loss[i_init][i_batch]}\n")
                f.write(
                    f"final rotation error =\
                         {all_init_end_rot_errors[i_init][i_batch]}\n"
                )
                f.write(
                    f"final translation error =\
                         {all_init_end_tra_errors[i_init][i_batch]}\n"
                )
        sublogdir = os.path.join(base_dir, str(i_batch + batch_start_idx))
        with open(os.path.join(sublogdir, "pose.txt"), "w", encoding="utf8") as f:
            f.write(f"target pose = {target_poses[i_batch]}\n")
            f.write(
                f"predicted pose = {all_init_camposes[chosen_inits[i_batch]][i_batch]}\n"
            )
            f.write(
                f"initial loss = {all_init_start_loss[chosen_inits[i_batch]][i_batch]}\n"
            )
            f.write(
                f"initial rotation error = \
                    {all_init_start_rot_errors[chosen_inits[i_batch]][i_batch]}\n"
            )
            f.write(
                f"initial translation error = \
                {all_init_start_tra_errors[chosen_inits[i_batch]][i_batch]}\n"
            )
            f.write(
                f"final iou = {all_init_end_ious[chosen_inits[i_batch]][i_batch]}\n"
            )
            f.write(
                f"final loss = {all_init_end_loss[chosen_inits[i_batch]][i_batch]}\n"
            )
            f.write(
                f"final rotation error = \
                    {all_init_end_rot_errors[chosen_inits[i_batch]][i_batch]}\n"
            )
            f.write(
                f"final translation error = \
                    {all_init_end_tra_errors[chosen_inits[i_batch]][i_batch]}\n"
            )
            f.write(f"chosen init idx = {chosen_inits[i_batch]}\n")
            f.write(f"best rot idx = {best_rot_inits[i_batch]}\n")
            f.write(f"best tra idx = {best_pos_inits[i_batch]}\n")
