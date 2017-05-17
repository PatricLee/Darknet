# Darknet - Velodyne

Modified darknet for velodyne, in Windows, using MS Visual Studio 2015, CUDA 8.0 and OpenCV 2.4.9

This repository is forked from Yolo-Windows v2: https://github.com/AlexeyAB/darknet

which is forked from the original Linux version: https://github.com/pjreddie/darknet

which you may check out at: https://pjreddie.com/darknet/yolo

I would like to thank AlexeyAB for his amazing job and his help

## To Install

Please follow installation procedure in Yolo-Windows v2: https://github.com/AlexeyAB/darknet

## Under Construction

This code is for my thesis, where I modify and use Yolo on Velodyne data to detect surrounding vehicles on our autonomous driving platform

For now the network looks promising, on a 304x304 input in bird view, which is 30.4x30.4 meter, the network is able to predict 2D or even 3D boxes with orientation estimation within 20ms on a single GTX1080. With a IoU threshold of .6, the network managed to reach 70% recall (on KITTI dataset).

### Note that this repository is still under construction

and would remain this way before my thesis is finished. Any usage, including usage strictlly following my instruction, may lead to unexpected behavior.

This repository is forked from Yolo-Windows v2, but with a ton of midifications. I'll try explaining each of them as they are also part of my thesis.

Issue/direct contact is welcomed.
