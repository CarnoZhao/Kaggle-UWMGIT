# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.core.seg.builder import build_pixel_sampler
from mmseg.models.utils.transunet import TransUnetEncoderWrapper
from mmseg.ops import resize
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from .base import BaseSegmentor
from ..losses import accuracy
from ..builder import SEGMENTORS, build_loss, build_backbone
from ..utils import TransUnetEncoderWrapper

import segmentation_models_pytorch as smp

class SwinWrapper(nn.Module):
    def __init__(self, encoder_name, in_channels = 3, weights = None):
        super(SwinWrapper, self).__init__()
        configs = {
            "swin_tiny": dict(
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                use_abs_pos_embed=False,
                drop_path_rate=0.3,
                patch_norm=True
            ),
            "swin_small": dict(
                embed_dims=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                use_abs_pos_embed=False,
                drop_path_rate=0.3,
                patch_norm=True
            ),
            "swin_base": dict(
                embed_dims=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=7,
                use_abs_pos_embed=False,
                drop_path_rate=0.3,
                patch_norm=True
            ),
        }
        out_channels = {
            "swin_tiny": [96, 192, 384, 768],
            "swin_small": [96, 192, 384, 768],
            "swin_base": [128, 256, 512, 1024]
        }
        config = configs[encoder_name]
        config["type"] = "SwinTransformer"
        config["in_channels"] = in_channels
        config["pretrained"] = weights
        if "384" in weights:
            config["window_size"] = 12
            config["pretrain_img_size"] = 384
        else:
            assert "224" in weights, Exception("need a weight path with valid pretrained image size")
        self.model = build_backbone(config)
        self.out_channels = [in_channels,] + out_channels[encoder_name]

    def forward(self, x):
        outs = [x]
        outs.extend(self.model(x))
        return outs

@SEGMENTORS.register_module()
class SMPUnet(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 ignore_index=255,
                 pretrained=None,
                 init_cfg=None):
        super(SMPUnet, self).__init__(init_cfg)
        encoder_name = backbone.get("type")
        in_channels = backbone.get("in_channels", 3)
        encoder_weights = backbone.get("pretrained", "imagenet")
        encoder_depth = backbone.get("depth", 5)
        decoder_channels = decode_head.get("channels", (256, 128, 64, 32, 16))
        decoder_use_batchnorm = decode_head.get("use_batchnorm", True)
        decoder_attention_type = decode_head.get("attention_type", None)
        classes = decode_head.get("num_classes", 1)
        activation = decode_head.get("activation", None)

        if "swin" in encoder_name:
            self.backbone = SwinWrapper(
                encoder_name,
                in_channels=in_channels,
                weights=encoder_weights,
            )
            encoder_depth = 4
        else:
            self.backbone = smp.encoders.get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

        transunet = backbone.get("transunet", None)
        if transunet is not None:
            self.backbone = TransUnetEncoderWrapper(self.backbone, transunet)

        self.decode_head = smp.unet.decoder.UnetDecoder(
            encoder_channels=self.backbone.out_channels,
            decoder_channels=decoder_channels[:encoder_depth],
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[:encoder_depth][-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )


        self.align_corners = decode_head.get("align_corners", 1)
        self.num_classes = decode_head.get("num_classes", 1)
        if decode_head.get("sampler", False):
            self.sampler = build_pixel_sampler(decode_head["sampler"], context=self)
        else:
            self.sampler = None
        self.ignore_index = ignore_index


        loss_decode = decode_head.get("loss_decode", None)
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        seg_logits = self._decode_head_forward_test(x, img_metas)
        loss_decode = self.losses(seg_logits, gt_semantic_seg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.segmentation_head(self.decode_head(*x))
        return seg_logits

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            if 'pad_shape' in img_meta[0] and img_meta[0]["pad_shape"] != img_meta[0]["img_shape"]:
                seg_logit = seg_logit[...,:img_meta[0]["img_shape"][0],:img_meta[0]["img_shape"][1]]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale, **kwargs):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)

        losses = {}
        if 'gt_semantic_seg' in kwargs:
            gt_semantic_seg = kwargs['gt_semantic_seg']
            loss_decode = self.losses(seg_logit, gt_semantic_seg)

            losses.update(add_prefix(loss_decode, 'decode'))
        if not self.test_cfg.get("multi_label", False):
            output = F.softmax(seg_logit, dim=1)
        else:
            output = F.sigmoid(seg_logit)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output, losses

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
        seg_logit, losses = self.inference(img, img_meta, rescale, **kwargs)
        if self.test_cfg.get("logits", False) and self.test_cfg.get("multi_label", False):
            seg_pred = seg_logit.permute(0, 2, 3, 1)# .argmax(dim=1)
        elif self.test_cfg.get("logits", False):
            seg_pred = seg_logit
        elif self.test_cfg.get("binary_thres", None) is not None:
            seg_pred = seg_logit[:,1] > self.test_cfg.get("binary_thres")
            seg_pred = seg_pred.long()
        elif self.test_cfg.get("multi_label", False):
            thres = self.test_cfg.get("multi_label_thres", 0.5)
            seg_pred = (seg_logit > thres).permute(0, 2, 3, 1)
            seg_pred = seg_pred.long()
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)

        return seg_pred, losses

    def aug_test(self, imgs, img_metas, rescale=True, **kwargs):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit, losses = self.inference(imgs[0], img_metas[0], rescale, **kwargs)
        for i in range(1, len(imgs)):
            cur_seg_logit, cur_loss = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
            for k in losses:
                losses[k] += cur_loss[k]
        seg_logit /= len(imgs)
        for k in losses:
            losses[k] /= len(imgs)
        if self.test_cfg.get("logits", False) and self.test_cfg.get("multi_label", False):
            seg_pred = seg_logit.permute(0, 2, 3, 1)# .argmax(dim=1)
        elif self.test_cfg.get("logits", False):
            seg_pred = seg_logit
        elif self.test_cfg.get("binary_thres", None) is not None:
            seg_pred = seg_logit[:,1] > self.test_cfg.get("binary_thres")
            seg_pred = seg_pred.long()
        elif self.test_cfg.get("multi_label", False):
            thres = self.test_cfg.get("multi_label_thres", 0.5)
            seg_pred = (seg_logit > thres).permute(0, 2, 3, 1)
            seg_pred = seg_pred.long()
        else:
            seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred, losses

@SEGMENTORS.register_module()
class SMPUnetPlusPlus(SMPUnet):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 ignore_index=255,
                 pretrained=None,
                 init_cfg=None):
        super(SMPUnetPlusPlus, self).__init__(backbone, decode_head, neck, auxiliary_head, train_cfg, test_cfg, ignore_index, pretrained, init_cfg)
        encoder_name = backbone.get("type")
        encoder_depth = backbone.get("depth", 5)
        decoder_channels = decode_head.get("channels", (256, 128, 64, 32, 16))
        decoder_use_batchnorm = decode_head.get("use_batchnorm", True)
        decoder_attention_type = decode_head.get("attention_type", None)

        self.decode_head = smp.unetplusplus.decoder.UnetPlusPlusDecoder(
            encoder_channels=self.backbone.out_channels,
            decoder_channels=decoder_channels[:encoder_depth],
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )