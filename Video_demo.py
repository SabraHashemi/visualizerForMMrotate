# Copyright (c) OpenMMLab. All rights reserved.
"""Inference on huge images.

Example:
```
wget -P checkpoint https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth  # noqa: E501, E261.
python demo/huge_image_demo.py \
    demo/dota_demo.jpg \
    configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_v3.py \
    checkpoint/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
```
"""  # nowq

from argparse import ArgumentParser

from mmdet.apis import init_detector, show_result_pyplot

from mmrotate.apis import inference_detector_by_patches
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from PIL import Image
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.1,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    cap = cv2.VideoCapture(args.video)
    while(cap.isOpened()):
        start=time.time()
        ret, frame = cap.read()
        new_width = int(frame.shape[1]/2)
        new_height = int(frame.shape[0]/2)

        frame = cv2.resize(frame, (new_width, new_height))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

    # test a huge image by patches
        #result = inference_detector_by_patches(model, frame, args.patch_sizes,args.patch_steps, args.img_ratios, args.merge_iou_thr)
        result = inference_detector(model, frame)
        end=time.time()
        fps=1/(end-start)
        frame = cv2.putText(frame, str(fps), org, font, fontScale, color, thickness, cv2.LINE_AA)
    # show the results
        vis_frame=model.show_result(
            frame, result, score_thr=args.score_thr, wait_time=1, show=False)
        
        cv2.imshow('pytorch_result', vis_frame)
        cv2.waitKey(1)




if __name__ == '__main__':
    args = parse_args()
    main(args)
