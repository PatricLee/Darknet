# Darknet - Velodyne

Modified darknet for Velodyne, in Windows, to detect surrounding vehicles on an autonomous driving platform.
This repository is forked from Yolo-Windows v2: https://github.com/AlexeyAB/darknet
which is forked from the original Linux version: https://github.com/pjreddie/darknet
which you may check out at: https://pjreddie.com/darknet/yolo
I would like to thank AlexeyAB for his amazing job as well as his help.

## 0 Requirement
- **Microsoft Visual Studio 2015**, or v14.0.
- **CUDA 8.0, with cuDNN 5.1 as optional**.
- **OpenCV 2.4.9**.
  - If you don't want to use OpenCV, simply remove define OPENCV in Visual Studio -> Project -> Properties -> C/C++ -> Preprocessor.
  - If you have another version of OpenCV, then you may have to change a lot of places, or simply disable it.
  - I use `.png` picture as it is not compressed in the way `.jpg` is, so it's totally fine if you don't use OpenCV.
    
### About the name
I named my part of the project as 'Cronus', you may change the name if you're not okay with it. Open `src/darknet.c`, `src/cronus.c`, `src/cronus.h`, find and replace all 'cronus' with a name that you like, e.g. Zeus. Then rename `src/cronus.c` and `src/cronus.h`, open `build/darknet/darknet.sln` and add two renamed files to the project. Note that all 'cronus' mentioned in later sections will be changed as well.

## 1 To Compile

### 1.1 If your environment meets the requirements above
Start MSVS, open `build/darknet/darknet.sln`, set **x64** and **Release**, build project. *In what world is life that easy*.

### 1.2 If yours does not

#### 1.2.1 You have other version of MS Visual Studio, which is not 2015 or v14.0
Then you have to create your own version of `darknet.sln` and `darknet.vcxproj`. I'll come back to explaining this some day later as this problem could be ***easily*** solved by installing VS 2015.

#### 1.2.2 You have other version of CUDA, which is not 8.0
- Open `build/darknet/darknet.vcxproj` with Notepad, find and replace **CUDA 8.0** with **CUDA x.x** (Your version). There are 2 of them.
- Compile as said in section 1.1.

This should work for CUDA 6.0 and above, I'm not sure what will happen if you use a even earlier version, or a later version that is yet to come. If you run into CUDA problems with version earlier than 7.5, I suggest updating your CUDA to 8.0.

#### 1.2.3 You don't want to use cuDNN
If you use CUDA 8.0 I strongly recommand using cuDNN 5.1. But if you don't want to, open `build/darknet/darknet.sln`, right click on project -> properties -> C/C++ -> Preprocessor -> Preprocessor Definations, find and remove the line `CUDNN`.

#### 1.2.4 You have other version of OpenCV, which is 2.4.x but not 2.4.9
- Open `build/darknet/darknet.sln` in VS, right click on project -> properties -> C/C++ -> General -> Additional Include Directories.
- Right click on project -> p[roperties -> Linker -> General -> Additional Library Directories.
- Open `src/cronus.c`, change `249` in following lines to your version, like `2413` for version 2.4.13.
  - `#pragma comment(lib, "opencv_core249.lib")`
  - `#pragma comment(lib, "opencv_imgproc249.lib")`
  - `#pragma comment(lib, "opencv_highgui249.lib")` 

#### 1.2.5 You have other version of OpenCV, which is not 2.4.x
Then you have to change A LOT of places in the code. *Why give yourself a hard time*?
- Open `build/darknet/darknet.sln`, find all '#ifdef OPENCV' in every file.
- Change all functions between **#ifdef OPENCV** and **#endif** to functions that do the same job in your version of OpenCV.

## 2 Configure the Network



## Under Construction

This code is for my thesis, where I modify and use Yolo on Velodyne data to detect surrounding vehicles on our autonomous driving platform

For now the network looks promising, on a 304x304 input in bird view, which is 30.4x30.4 meter, the network is able to predict 2D or even 3D boxes with orientation estimation within 20ms on a single GTX1080. With a IoU threshold of .6, the network managed to reach 70% recall (on KITTI dataset).

### Note that this repository is still under construction

and would remain this way before my thesis is finished. Any usage, including usage strictlly following my instruction, may lead to unexpected behavior.

This repository is forked from Yolo-Windows v2, but with a ton of midifications. I'll try explaining each of them as they are also part of my thesis.

Issue/direct contact is welcomed.
