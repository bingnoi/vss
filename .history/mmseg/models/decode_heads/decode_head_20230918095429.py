from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy
import torch.nn.functional as F

import numpy as np

from .criterion import SetCriterion
from .matcher import HungarianMatcher

class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        # print(seg_label.shape, seg_label.min(), seg_label.max())
        # exit()
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
        # print(seg_label.shape, seg_logit.shape)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        # print("here: ",loss)
        # exit()
        return loss


class BaseDecodeHead_clips(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead_clips.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 num_clips=5,
                 hypercorre=False,
                 cityscape=False,
                 backbone='b1'):
        super(BaseDecodeHead_clips, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.num_clips=num_clips

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False
        self.hypercorre=hypercorre
        self.atten_loss=False
        self.self_ensemble2=False
        self.cityscape=cityscape
        self.backbone=backbone

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg,batch_size, num_clips):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs,batch_size, num_clips)
        losses = self.losses(seg_logits, inputs,gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, batch_size=None, num_clips=None):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, batch_size, num_clips)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def _construct_ideal_affinity_matrix(self, label, label_size):
        assert label.dim()==5
        B,num_clips,c,h_label,w_label=label.shape
        assert c==1
        label=label.reshape(B*num_clips,c,h_label,w_label)
        scaled_labels = F.interpolate(
            label.float(), size=label_size, mode="nearest")
        scaled_labels = scaled_labels.squeeze_().long()
        scaled_labels[scaled_labels == 255] = self.num_classes
        # print("here: ",scaled_labels.shape)
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
        one_hot_labels = one_hot_labels.view(
            B, num_clips, -1, self.num_classes + 1).float()
        one_hot_labels_lastframe=one_hot_labels[:,-1:]
        one_hot_labels_reference=one_hot_labels[:,:-1]
        # ideal_affinity_matrix = torch.bmm(one_hot_labels,
        #                                   one_hot_labels.permute(0, 2, 1))
        ideal_affinity_matrix=torch.matmul(one_hot_labels_lastframe,
                                           one_hot_labels_reference.transpose(-2, -1))
        assert ideal_affinity_matrix.dim()==4
        return ideal_affinity_matrix.reshape(B*(num_clips-1), ideal_affinity_matrix.shape[-2], ideal_affinity_matrix.shape[-1])

    def prepare_targets(self, targets, images):
        h, w = 480,480
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image['gt_masks']
            
            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            # print('mask',gt_masks.shape,padded_masks.shape)
            try:
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            except:
                print("sss",padded_masks.shape,gt_masks.shape)

            # print('ffff',targets_per_image['gt_classes'].device,padded_masks.device)
            new_targets.append(
                {
                    "labels": targets_per_image['gt_classes'],
                    "masks": padded_masks,
                }
            )
        return new_targets

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, images,seg_label):
        """Compute segmentation loss."""
        
        ins_data = []
        split_group = torch.split(seg_label,split_size_or_sections=1,dim=1)
        # print("ssk",seg_label.shape,[i.shape for i in split_group])
        for i_label in split_group:
            sem_seg_gt = i_label
            sem_seg = sem_seg_gt.cpu().numpy()
            classes = np.unique(sem_seg)
            classes = classes[classes != self.ignore_index]
            
            gt_classes = torch.tensor(classes,dtype=torch.int64)
            masks = []
            for class_id in classes:
                masks.append(sem_seg == class_id)
                print("see",class_id,torch.count_nonzero(sem_seg == class_id))
            exit()
            if len(masks) == 0:
                gt_mask = torch.zeros((0, sem_seg.shape[-2], sem_seg.shape[-1]))
            else:
                gt_mask = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
        
            ins = {}
            
            gt_classes = gt_classes.to('cuda' if torch.cuda.is_available() else 'cpu')
            gt_mask = gt_mask.to('cuda' if torch.cuda.is_available() else 'cpu').squeeze()
            
            if len(gt_mask.shape) < 3:
                gt_mask=gt_mask.unsqueeze(0)
            
            ins['gt_classes'] = gt_classes
            ins['gt_masks'] = gt_mask
            
            
            ins_data.append(ins)
            
        mask_weight = 20
        dice_weight = 1
        deep_supervision = True
        no_object_weight =0.1
        
        matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )
        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = 6
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        losses = ["labels", "masks"]
        self.criterion = SetCriterion(
            111,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
        )

        # print("shape",[i.shape for i in images])
        
        targets = self.prepare_targets(ins_data,images)
        
        # seg_logit['pred_logits'] = seg_logit['pred_logits'].to('cpu')
        # seg_logit['pred_masks']=seg_logit['pred_masks'].to('cpu')
        losses = self.criterion(seg_logit,targets)
        
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses
        
        
        # torch.Size([1, 8, 124, 120, 120]) torch.Size([1, 4, 1, 480, 480])

        # assert seg_logit.dim()==5 and seg_label.dim()==5

        # loss = dict()

        # if self.hypercorre and self.cityscape:
        #     # print("here1")
        #     assert seg_logit.shape[1]==2*seg_label.shape[1]
        #     num_clips=seg_label.shape[1]
        #     seg_logit_ori=seg_logit[:,num_clips-1:num_clips]
        #     batch_size, _, _, h ,w=seg_logit_ori.shape
        #     seg_logit_ori=seg_logit_ori.reshape(batch_size,-1,h,w)
        #     seg_logit_lastframe=seg_logit[:,num_clips:].reshape(batch_size*(num_clips),-1,h,w)

        #     batch_size, num_clips, _, h ,w=seg_label.shape
        #     seg_label_ori=seg_label[:,-1]
        #     # print(seg_label[:,-1].shape)
        #     seg_label_lastframe=seg_label[:,-1:].expand(batch_size,num_clips,1,h,w).reshape(batch_size*(num_clips),1,h,w)
        # elif self.hypercorre:
        #     # print("here2")
        #     # print(self.self_ensemble2,seg_logit.shape[1],seg_label.shape[1])
        #     # if self.self_ensemble2 and seg_logit.shape[1]==2*seg_label.shape[1]:
        #     if self.self_ensemble2 and seg_logit.shape[1]==seg_label.shape[1] + 4:

        #         # assert seg_logit.shape[1]==2*seg_label.shape[1]
        #         num_clips=seg_label.shape[1]
        #         num_clips_last = 4
        #         seg_logit_ori=seg_logit[:,:num_clips]
        #         batch_size, _, _, h ,w=seg_logit_ori.shape

        #         seg_logit_ori=seg_logit_ori.reshape(batch_size*(num_clips),-1,h,w)
        #         seg_logit_last3frame=seg_logit[:,num_clips:-1].reshape(batch_size*(num_clips_last-1),-1,h,w)
        #         seg_logit_lastframe = seg_logit[:,-1:].reshape(batch_size,-1,h,w)

        #         batch_size, num_clips, chan, h ,w=seg_label.shape

        #         assert chan==1
        #         seg_label_ori=seg_label.reshape(batch_size*(num_clips),1,h,w)

        #         # seg_label_lastframe=seg_label[:,-1:].expand(batch_size,num_clips,1,h,w).reshape(batch_size*(num_clips),1,h,w)
        #         # seg_label_lastframe=seg_label[:,-1:].expand(batch_size,num_clips_last,1,h,w).reshape(batch_size*(num_clips_last),1,h,w)

        #         # seg_label_lastframe=seg_label[:,-1:].expand(batch_size,num_clips_last,1,h,w).reshape(batch_size*(num_clips_last),1,h,w)
        #         seg_label_last3frame = seg_label[:,:-1].reshape(batch_size*(num_clips_last-1),-1,h,w)


        #         seg_label_lastframe = seg_label[:,-1:].reshape(batch_size,-1,h,w)


        #     else:
        #         assert False, "parameters not correct"            

        # # print('l',seg_logit_ori.shape, seg_logit_lastframe.shape)

        # # print('m',seg_label_ori.shape, seg_label_lastframe.shape)

        # # 一个是x,剩下那个是out1...4
        # seg_logit_ori = resize(
        #     input=seg_logit_ori,
        #     size=seg_label.shape[3:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        # seg_logit_last3frame = resize(
        #     input=seg_logit_last3frame,
        #     size=seg_label.shape[3:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        # seg_logit_lastframe = resize(
        #     input=seg_logit_lastframe,
        #     size=seg_label.shape[3:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        # if self.sampler is not None:
        #     seg_weight = self.sampler.sample(seg_logit, seg_label)
        # else:
        #     seg_weight = None

        # seg_label_ori = seg_label_ori.squeeze(1)
        # seg_label_last3frame = seg_label_last3frame.squeeze(1)
        # seg_label_lastframe = seg_label_lastframe.squeeze(1)

        

        # if(not (np.unique(seg_label_ori.cpu()) == np.unique(seg_label_ori.cpu())).any()):
        #     print(np.unique(seg_label_ori.cpu()),np.unique(seg_label_ori.cpu()))
        #     exit()

        # loss['loss_seg'] = self.loss_decode(
        #     seg_logit_ori,
        #     seg_label_ori,
        #     weight=seg_weight,
        #     ignore_index=self.ignore_index)+self.loss_decode(
        #     seg_logit_last3frame,
        #     seg_label_last3frame,
        #     weight=seg_weight,
        #     ignore_index=self.ignore_index)+self.loss_decode(
        #     seg_logit_lastframe,
        #     seg_label_lastframe,
        #     weight=seg_weight,
        #     ignore_index=self.ignore_index
        #     )

        # loss['acc_seg'] = accuracy(seg_logit_ori, seg_label_ori)

        
    