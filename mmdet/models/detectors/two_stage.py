import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        if 0:
            import numpy as np
            import os
            import cv2
            bbox_results = proposal_list[0].detach().cpu().numpy()
            polys = np.array(bbox_results).reshape(-1, 5)
            image_source = img.cpu().detach().numpy()
            image_source = np.transpose(image_source, (0, 2, 3, 1))
            mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1)
            std = np.array([58.395, 57.12, 57.375]).reshape(1, -1)
            image_source = np.multiply(image_source, std)
            image_source = np.add(image_source, mean).astype(np.uint8)
            image_source = image_source[0].copy()
            # image_dir = img_metas[0]['filename']
            # image_source = cv2.imread(image_dir)
            for id in range(polys.shape[0]):
                x1 = polys[id][0]
                y1 = polys[id][1]
                x2 = polys[id][2]
                y2 = polys[id][3]
                cv2.rectangle(image_source, (x1, y1), (x2, y2), (0, 255, 0), 1)

            cv2.namedWindow("AlanWang")
            cv2.imshow('AlanWang', image_source)
            cv2.waitKey(10)  # 显示 10000 ms 即 10s 后消失
            cv2.destroyAllWindows()

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)
        outs = self.rpn_head(x)

        if proposals is None:
            proposal_list = self.rpn_head.get_bboxes(*outs, img_metas)
        else:
            proposal_list = proposals
        if 1:
            import numpy as np
            import os
            bbox_results = proposal_list[0].detach().cpu().numpy()
            polys = np.array(bbox_results).reshape(-1, 5)
            image_jjj = img_metas[0]['ori_filename'][0:-4]
            scale_factor = img_metas[0]['scale_factor']
            with open('{}'.format(os.path.join('evaluate/icdar2015_evalu/res_fcos', 'res_{}.txt'.format(image_jjj))), 'w') as f:
                for id in range(polys.shape[0]):
                    x1 = polys[id][0] / scale_factor[0]
                    y1 = polys[id][1] / scale_factor[1]
                    x2 = polys[id][2] / scale_factor[2]
                    y2 = polys[id][3] / scale_factor[3]
                    f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                        int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)))
        if 1:
            import numpy as np
            import cv2
            bbox_results = proposal_list[0].detach().cpu().numpy()
            polys = np.array(bbox_results).reshape(-1, 5)
            image_source = img.cpu().detach().numpy()
            image_source = np.transpose(image_source, (0, 2, 3, 1))
            mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1)
            std = np.array([58.395, 57.12, 57.375]).reshape(1, -1)
            image_source = np.multiply(image_source, std)
            image_source = np.add(image_source, mean).astype(np.uint8)
            image_source = image_source[0].copy()
            # image_dir = img_metas[0]['filename']
            # image_source = cv2.imread(image_dir)
            for id in range(polys.shape[0]):
                x1 = polys[id][0]
                y1 = polys[id][1]
                x2 = polys[id][2]
                y2 = polys[id][3]
                cv2.rectangle(image_source, (x1, y1), (x2, y2), (0, 255, 0), 1)

            cv2.imwrite('evaluate/proposal_img/'+img_metas[0]['ori_filename'], image_source)
            # cv2.namedWindow("AlanWang")
            # cv2.imshow('AlanWang', image_source)
            # cv2.waitKey(10)  # 显示 10000 ms 即 10s 后消失
            # cv2.destroyAllWindows()

        results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
        if 1:
            import numpy as np
            import os
            bbox_results = results[0][0][0]
            polys = np.array(bbox_results).reshape(-1, 5)
            image_jjj = img_metas[0]['ori_filename'][0:-4]
            with open('{}'.format(os.path.join('evaluate/icdar2015_evalu/res', 'res_{}.txt'.format(image_jjj))), 'w') as f:
                for id in range(polys.shape[0]):
                    x1 = polys[id][0]
                    y1 = polys[id][1]
                    x2 = polys[id][2]
                    y2 = polys[id][3]
                    f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                        int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)))

        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
