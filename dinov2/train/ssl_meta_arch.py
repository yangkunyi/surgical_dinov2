# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import math
from functools import partial

import torch
from dinov2.fsdp import (
    ShardedGradScaler,
    get_fsdp_modules,
    get_fsdp_wrapper,
    reshard_fsdp_model,
)
from dinov2.layers import DINOHead
from dinov2.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from dinov2.models import build_model_from_cfg

from dinov2.models.vision_transformer import BlockChunk
from dinov2.utils.param_groups import fuse_params_groups, get_params_groups_with_decay
from dinov2.utils.utils import has_batchnorms
from torch import nn
import torch.nn.functional as F
import sys

try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


def segmentation_loss(raw_image_patch_tokens, segmentation, segmentation_temperature):
    """
    Computes the segmentation supervision loss based on pairwise similarities *within* each sample.

    Args:
        raw_image_patch_tokens (torch.Tensor): Shape [B, N, D].
                                              Patch tokens (features) from the student model.
                                              N = number of patches, e.g., (H/14)*(W/14).
                                              D = feature dimension.
        segmentation (torch.Tensor): Shape [B, 1, H, W]. Original high-resolution segmentation maps.
        segmentation_temperature (float): Sigma (σ) value for the Gaussian kernel controlling
                                          segmentation similarity bandwidth.

    Returns:
        torch.Tensor: Scalar segmentation supervision loss (L_seg), averaged over the batch.
    """
    B, N_tokens, D_tokens = raw_image_patch_tokens.shape
    _B_seg, _C_seg, H_img, W_img = segmentation.shape
    device = raw_image_patch_tokens.device

    # Initial checks for valid inputs
    if B == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if N_tokens <= 1:  # Not enough tokens to form pairs for comparison
        return torch.tensor(0.0, device=device, requires_grad=True)

    if segmentation_temperature <= 1e-6:  # Avoid division by zero or very small numbers
        segmentation_temperature = 1e-6

    # 1. Downsample segmentation using mode pooling (batched)
    assumed_patch_size = 14
    if H_img % assumed_patch_size != 0 or W_img % assumed_patch_size != 0:
        raise ValueError(
            f"Image dimensions H_img={H_img}, W_img={W_img} are not divisible by "
            f"assumed_patch_size={assumed_patch_size}."
        )
    num_patches_h = H_img // assumed_patch_size
    num_patches_w = W_img // assumed_patch_size
    if num_patches_h * num_patches_w != N_tokens:
        raise ValueError(
            f"N_tokens ({N_tokens}) does not match the expected number of patches "
            f"({num_patches_h * num_patches_w}) derived from H_img={H_img}, W_img={W_img} "
            f"and assumed_patch_size={assumed_patch_size}."
        )

    seg_unfold = F.unfold(
        segmentation.float(), kernel_size=assumed_patch_size, stride=assumed_patch_size
    )
    if seg_unfold.shape[2] != N_tokens:
        raise AssertionError(
            f"Mismatch after unfold: seg_unfold.shape[2]={seg_unfold.shape[2]}, "
            f"expected N_tokens={N_tokens}"
        )
    mode_values, _ = torch.mode(seg_unfold, dim=1)
    seg_labels_flat = mode_values.to(torch.int8)  # Shape: [B, N_tokens]

    # 2. Normalize patch tokens (batched)
    tokens_norm = F.normalize(
        raw_image_patch_tokens, p=2, dim=2
    )  # Shape [B, N_tokens, D_tokens]

    # 3. Compute pairwise cosine similarities (batched)
    S_ij_batch = torch.bmm(
        tokens_norm, tokens_norm.transpose(1, 2)
    )  # Shape: [B, N_tokens, N_tokens]
    S_ij_scaled_batch = (S_ij_batch + 1.0) / 2.0
    # 4. Create batched target similarity matrix
    target_matrix_batch = (
        seg_labels_flat.unsqueeze(2) == seg_labels_flat.unsqueeze(1)
    ).float()  # Shape: [B, N_tokens, N_tokens]

    # 5. Compute loss per pair (batched)
    logits_batch = S_ij_scaled_batch / segmentation_temperature
    loss_all_pairs_batch = F.binary_cross_entropy_with_logits(
        logits_batch, target_matrix_batch, reduction="none"
    )  # Shape: [B, N_tokens, N_tokens]

    # 6. Create masks for valid pairs to be included in the loss
    # 6a. Mask for off-diagonal elements (i != j)
    mask_off_diagonal = ~torch.eye(
        N_tokens, dtype=torch.bool, device=device
    )  # Shape: [N_tokens, N_tokens]

    # 6b. Mask for tokens that are segmented (label != 0)
    token_is_segmented_mask = seg_labels_flat != 0  # Shape: [B, N_tokens]

    # 6c. Mask for pairs where BOTH tokens in the pair (i, j) are segmented
    # token_is_segmented_mask.unsqueeze(2) -> [B, N_tokens, 1]
    # token_is_segmented_mask.unsqueeze(1) -> [B, 1, N_tokens]
    # Broadcasting results in [B, N_tokens, N_tokens] where M[b,i,j] is true if token i AND token j in sample b are segmented
    pair_both_segmented_mask = token_is_segmented_mask.unsqueeze(
        2
    ) & token_is_segmented_mask.unsqueeze(1)

    # 6d. Final combined mask: pair must be off-diagonal AND both tokens in the pair must be segmented.
    # Unsqueeze mask_off_diagonal to broadcast it from [N_tokens, N_tokens] to [1, N_tokens, N_tokens] for batch operation.
    final_active_pair_mask = pair_both_segmented_mask & mask_off_diagonal.unsqueeze(
        0
    )  # Shape: [B, N_tokens, N_tokens]

    # 7. Calculate "mean of sample means" using the final_active_pair_mask

    # Sum of losses for active pairs per sample. Inactive pairs (masked out) will contribute 0 to this sum.
    # Multiplying by final_active_pair_mask.float() effectively zeroes out losses from inactive pairs.
    active_losses_sum_per_sample = (
        loss_all_pairs_batch * final_active_pair_mask.float()
    ).sum(dim=[1, 2])

    # Number of active (valid) pairs per sample
    num_active_pairs_per_sample = final_active_pair_mask.sum(
        dim=[1, 2]
    )  # Summing boolean tensor gives count of True values

    # Identify samples that have at least one active pair
    samples_with_valid_pairs_mask = num_active_pairs_per_sample > 0

    # Initialize sample_mean_losses tensor (e.g., with zeros)
    sample_mean_losses = torch.zeros(B, device=device, dtype=loss_all_pairs_batch.dtype)

    # Calculate mean loss only for samples that have active pairs to avoid division by zero
    if samples_with_valid_pairs_mask.any():  # Check if there's at least one such sample
        # Get the sums and counts only for the samples that have valid pairs
        valid_sums = active_losses_sum_per_sample[samples_with_valid_pairs_mask]
        valid_counts = num_active_pairs_per_sample[samples_with_valid_pairs_mask]

        sample_mean_losses[samples_with_valid_pairs_mask] = valid_sums / valid_counts

    # Final loss: mean of the mean_losses from samples that had valid (active) pairs.
    # If no sample had any valid pair after all masking, the loss is 0.
    if samples_with_valid_pairs_mask.any():
        final_loss = sample_mean_losses[samples_with_valid_pairs_mask].mean()
    else:
        final_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return final_loss


def depth_loss(raw_image_patch_tokens, depths, depth_temperature):
    """
    Computes the depth supervision loss based on pairwise similarities *within* each sample.

    Args:
        raw_image_patch_tokens (torch.Tensor): Shape [B, N, D].
                                              Patch tokens (features) from the student model.
                                              N = number of patches, e.g., (H/14)*(W/14).
                                              D = feature dimension.
        depths (torch.Tensor): Shape [B, 1, H, W]. Original high-resolution depth maps.
        depth_temperature (float): Sigma (σ) value for the Gaussian kernel controlling
                                   depth similarity bandwidth.

    Returns:
        torch.Tensor: Scalar depth supervision loss (L_depth), averaged over the batch.
    """
    B, _, H, W = depths.shape
    _B_tokens, N, D = raw_image_patch_tokens.shape  # N = (H/patch_size)*(W/patch_size)

    # --- Infer Patch Size (same as before) ---
    patch_size_h = H / (N**0.5)
    patch_size_w = W / (N**0.5)
    if H % 14 == 0 and W % 14 == 0 and N == (H // 14) * (W // 14):
        patch_size = 14
    elif H % 16 == 0 and W % 16 == 0 and N == (H // 16) * (W // 16):
        patch_size = 16
    else:
        patch_size = int((H * W / N) ** 0.5)
        print(
            f"Warning: Could not perfectly infer patch size. Using approx: {patch_size}"
        )
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"Inferred patch size {patch_size} does not divide H={H} or W={W}"
            )

    # --- 1. Downsample Depths (same as before) ---
    downsampled_depths = F.avg_pool2d(depths, kernel_size=patch_size, stride=patch_size)
    # Shape: [B, 1, N_h, N_w]

    # Flatten the spatial dimensions for each sample independently
    # Shape becomes [B, N]
    downsampled_depths_flat_per_batch = downsampled_depths.view(B, -1)

    # --- 2. Calculate Pairwise Depth Similarity (T_ij) - BATCH-WISE ---
    # We want a result of shape [B, N, N] where T_ij_batch[b, i, j] = similarity between patch i and j of sample b.
    # Use broadcasting within the batch dimension:
    # depths [B, N] -> unsqueeze(2) -> [B, N, 1]
    # depths [B, N] -> unsqueeze(1) -> [B, 1, N]
    # Broadcasting subtraction [B, N, 1] - [B, 1, N] gives pairwise diffs [B, N, N]
    delta_ij_batch = torch.abs(
        downsampled_depths_flat_per_batch.unsqueeze(2)
        - downsampled_depths_flat_per_batch.unsqueeze(1)
    )  # Shape [B, N, N]

    # Apply Gaussian kernel element-wise
    # Add a small epsilon for numerical stability if depth_temperature is zero
    T_ij_batch = torch.exp(
        -delta_ij_batch / (depth_temperature + 1e-6)
    )  # Shape [B, N, N]

    # --- 3. Calculate Feature Similarity (S_ij_scaled) - BATCH-WISE ---
    # We want a result of shape [B, N, N] where S_ij_batch[b, i, j] = similarity between token i and j of sample b.
    # Tokens are already [B, N, D]

    # L2-normalize the features along the feature dimension (D)
    norm_tokens = F.normalize(raw_image_patch_tokens, p=2, dim=2)  # Shape [B, N, D]

    # Compute pairwise cosine similarities using Batch Matrix Multiplication (bmm)
    # We need [B, N, D] @ [B, D, N] -> [B, N, N]
    S_ij_batch = torch.bmm(norm_tokens, norm_tokens.transpose(1, 2))  # Shape [B, N, N]

    # Scale similarities to [0, 1]
    S_ij_scaled_batch = (S_ij_batch + 1.0) / 2.0  # Shape [B, N, N]

    # --- 4. Compute Depth Supervision Loss (L_depth) ---
    # Use Mean Squared Error (MSE). F.mse_loss computes the mean over all elements by default.
    # This will average the loss over all B*N*N pairs correctly.
    # We compare the similarity matrices for each sample element-wise.
    loss = F.mse_loss(S_ij_scaled_batch, T_ij_batch)

    return loss


# Example Usage (assuming you have the inputs):
# B, H, W, D = 4, 224, 224, 1024
# patch_size = 14
# N = (H // patch_size) * (W // patch_size) # N = 16 * 16 = 256
# dummy_tokens = torch.randn(B, N, D)
# dummy_depths = torch.rand(B, 1, H, W) * 10 # Example depth values
# temperature = 0.1 # Example sigma value

# loss_value = depth_loss(dummy_tokens, dummy_depths, temperature)
# print(f"Calculated Depth Loss: {loss_value.item()}")


def smooth_rank_loss(embedding_matrix, eps=1e-7):
    """
    Compute a loss based on the smooth rank measure of a matrix of embeddings.
    This version is adapted for use as a loss function, where lower values are better.

    Args:
        embedding_matrix (torch.Tensor): Matrix of embeddings (n x m). n: number of patch embeddings, m: embedding dimension.

    Returns:
        torch.Tensor: A scalar tensor representing the loss.
    """
    # Ensure the embeddings are float type for SVD
    embedding_matrix = embedding_matrix.float()

    # Perform SVD on the embedding matrix
    _, S, _ = torch.svd(embedding_matrix)

    # Normalize the singular values to sum to 1, add eps to avoid division by zero in log

    p = S / (torch.norm(S, p=1) + eps)
    p = p[: embedding_matrix.shape[1]]
    # Compute the negative entropy of the distribution
    # This encourages the concentration of information (lower entropy is better for loss minimization)
    neg_entropy = -torch.exp(-torch.sum(p * torch.log(p)))  # Add eps to avoid log(0)

    # Return the negative entropy as the loss
    return neg_entropy


def interpolate_pos_encoding(x, w, h):
    N = x.shape[1] - 1
    dim = x.shape[-1]
    w0 = w / int(math.sqrt(N))
    h0 = h / int(math.sqrt(N))

    # Interpolate the position embeddings without changing the first row (class token)
    patch_pos_embed = nn.functional.interpolate(
        x[:, 1:]
        .reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)
        .permute(0, 3, 1, 2),
        scale_factor=(w0, h0),
        mode="bicubic",
    )

    # assert int(w0) == patch_pos_embed.shape[-2]
    # assert int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    # Concatenate the class token with the interpolated position embeddings
    return torch.cat((x[:, :1], patch_pos_embed), dim=1)


def get_downloaded_dino_vit_interpolated(modelname="dinov2_vits14"):
    model = torch.hub.load("facebookresearch/dinov2", modelname, pretrained=True)  #
    input_tensor = model.pos_embed
    tensor_corr_shape = interpolate_pos_encoding(input_tensor, 16, 16)
    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed
    return model


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = (
            ShardedGradScaler() if cfg.compute_precision.grad_scaler else None
        )

        student_model_dict = dict()
        teacher_model_dict = dict()

        if cfg.student.arch in [
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ]:
            student_backbone = get_downloaded_dino_vit_interpolated(cfg.student.arch)
            teacher_backbone = get_downloaded_dino_vit_interpolated(cfg.student.arch)
            embed_dict = {
                "dinov2_vits14": 384,
                "dinov2_vitb14": 768,
                "dinov2_vitl14": 1024,
                "dinov2_vitg14": 1536,
            }
            embed_dim = embed_dict[cfg.student.arch]
        else:
            student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)

        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(
                f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}"
            )
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.do_depth_loss = cfg.depth.loss_weight > 0
        self.do_segmentation_loss = cfg.segmentation.loss_weight > 0

        self.do_smooth_rank_loss = cfg.dino.smooth_rank_loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DEPTH")
        if self.do_depth_loss:
            logger.info(f"OPTIONS -- DEPTH -- loss_weight: {cfg.depth.loss_weight}")
            logger.info(f"OPTIONS -- DEPTH -- temperature: {cfg.depth.temperature}")
            self.depth_loss_weight = cfg.depth.loss_weight
            self.depth_temperature = cfg.depth.temperature

        logger.info("OPTIONS -- SEGMENTATION")
        if self.do_segmentation_loss:
            logger.info(
                f"OPTIONS -- SEGMENTATION -- loss_weight: {cfg.segmentation.loss_weight}"
            )
            logger.info(
                f"OPTIONS -- SEGMENTATION -- temperature: {cfg.segmentation.temperature}"
            )
            self.segmentation_loss_weight = cfg.segmentation.loss_weight
            self.segmentation_temperature = cfg.segmentation.temperature

        logger.info("OPTIONS -- DINO")

        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(
                f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}"
            )
            logger.info(
                f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}"
            )
            logger.info(
                f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}"
            )
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(
            f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}"
        )
        logger.info(
            f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}"
        )
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, (
                "please provide a positive mask ratio tuple for ibot"
            )
            assert cfg.ibot.mask_sample_probability > 0, (
                "please provide a positive mask probability for ibot"
            )
            self.ibot_out_dim = (
                cfg.ibot.head_n_prototypes
                if self.ibot_separate_head
                else cfg.dino.head_n_prototypes
            )
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(
                    f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}"
                )
                logger.info(
                    f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}"
                )
                logger.info(
                    f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}"
                )
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(
            f"Student and Teacher are built: they are both {cfg.student.arch} network."
        )

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        raw_images = images["collated_raw_image"].cuda(non_blocking=True)
        depths = images["collated_depth"].cuda(non_blocking=True)
        segmentations = images["collated_segmentation"].cuda(non_blocking=True)
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat(
                (teacher_cls_tokens[1], teacher_cls_tokens[0])
            )
            ibot_teacher_patch_tokens = teacher_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                    upperbound + n_cls_tokens, _dim
                )
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[
                        n_cls_tokens : n_cls_tokens + n_masked_patches
                    ],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                    upperbound, _dim
                )
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(
                    teacher_cls_tokens
                )
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(
                    buffer_tensor_teacher
                )[:n_masked_patches]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(
                    teacher_cls_tokens
                )
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = (
                    self.dino_loss.softmax_center_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(
                        n_global_crops_teacher,
                        -1,
                        *teacher_cls_tokens_after_head.shape[1:],
                    )
                )
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = (
                        masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    )
                    masked_teacher_ibot_softmaxed_centered = (
                        self.ibot_patch_loss.softmax_center_teacher(
                            masked_teacher_patch_tokens_after_head[
                                :, :n_masked_patches
                            ],
                            teacher_temp=teacher_temp,
                        )
                    )
                    masked_teacher_ibot_softmaxed_centered = (
                        masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    )
                    self.ibot_patch_loss.update_center(
                        masked_teacher_patch_tokens_after_head[:n_masked_patches]
                    )

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = (
                    self.dino_loss.sinkhorn_knopp_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(
                        n_global_crops_teacher,
                        -1,
                        *teacher_cls_tokens_after_head.shape[1:],
                    )
                )

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = (
                        self.ibot_patch_loss.sinkhorn_knopp_teacher(
                            masked_teacher_patch_tokens_after_head,
                            teacher_temp=teacher_temp,
                            n_masked_patches_tensor=n_masked_patches_tensor,
                        )
                    )

            else:
                raise NotImplementedError

            return (
                teacher_dino_softmaxed_centered_list,
                masked_teacher_ibot_softmaxed_centered,
            )

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = (
            get_teacher_output()
        )
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = (
            self.student.backbone(
                [global_crops, local_crops], masks=[masks, None], is_training=True
            )
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict[
            "x_norm_clstoken"
        ]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(
                upperbound, _dim
            )
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(
                    ibot_student_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(
                    buffer_tensor_patch_tokens.unsqueeze(0)
                )
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(
                    buffer_tensor_patch_tokens
                )[:n_masked_patches]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(
            inputs_for_student_head_list
        )
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(
                0
            )[:n_masked_patches]

        if n_local_crops > 1:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(
                    n_local_crops
                ),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

            if self.do_smooth_rank_loss:
                smooth_rank_l = (
                    smooth_rank_loss(student_cls_tokens)
                    * self.cfg.dino.smooth_rank_loss_weight
                )
                loss_accumulator += smooth_rank_l
                loss_dict["smooth_rank_loss"] = smooth_rank_l / loss_scales

            if self.do_depth_loss or self.do_segmentation_loss:
                raw_image_patch_tokens = self.student.backbone(
                    raw_images, is_training=True
                )["x_norm_patchtokens"]
                if self.do_depth_loss:
                    depth_l = (
                        depth_loss(
                            raw_image_patch_tokens, depths, self.depth_temperature
                        )
                        * self.depth_loss_weight
                    )
                    loss_accumulator += depth_l
                    loss_dict["depth_loss"] = depth_l
                if self.do_segmentation_loss:
                    segmentation_l = (
                        segmentation_loss(
                            raw_image_patch_tokens,
                            segmentations,
                            self.segmentation_temperature,
                        )
                        * self.segmentation_loss_weight
                    )
                    loss_accumulator += segmentation_l
                    loss_dict["segmentation_loss"] = segmentation_l
                # print(raw_image_patch_tokens.shape, depths.shape)
                # print(self.depth_loss_weight)
                # print(self.depth_temperature)
                # sys.exit()

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()
        return loss_dict, student_cls_tokens

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = self.teacher.dino_head._streams = (
                self.student.backbone._streams
            ) = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(
                    get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])
                ):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(
                student_model_cfg, modules_to_wrap={BlockChunk}
            )(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(
                teacher_model_cfg, modules_to_wrap={BlockChunk}
            )(self.teacher[k])

    @staticmethod
    def interpolate_pos_encoding(x, w, h):
        # from Benedikt Roth, adapted from interpolate_pos_encoding in dinov2/dinov2/models/vision_transformer.py
        N = x.shape[1] - 1
        dim = x.shape[-1]
        w0 = w / int(math.sqrt(N))
        h0 = h / int(math.sqrt(N))

        # Interpolate the position embeddings without changing the first row (class token)
        patch_pos_embed = nn.functional.interpolate(
            x[:, 1:]
            .reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)
            .permute(0, 3, 1, 2),
            scale_factor=(w0, h0),
            mode="bicubic",
        )

        # assert int(w0) == patch_pos_embed.shape[-2]
        # assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # Concatenate the class token with the interpolated position embeddings
        return torch.cat((x[:, :1], patch_pos_embed), dim=1)
