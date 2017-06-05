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

## 1 Compile

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
If you use CUDA 8.0 I strongly recommand using cuDNN 5.1. But in case you don't want to,
- open `build/darknet/darknet.sln`, right click on project -> properties -> C/C++ -> Preprocessor -> Preprocessor Definations.
- Find and remove the line `CUDNN`.

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

## 2 Prepare input data

## 2.1 Velodyne data as picture

### 2.1.1 Channels of the picture
'Cronus' network processes LiDar data in Bird View, or the ground plane, so a 3D point cloud should first be projected into the ground grid. Each grid has a width and height of 0.1m, and has three channels:
- Red channel is height, or that of the highest point in the grid. I used a height range from -4m to 1m, mapped it to integer from 0 to 255 as the grayscale.
- Green channel is reflectance, or that of the point with highest reflectance in the grid. The reflectance in a Velodyne ranges from 0 to 1 in float number, so I mapped it to a 0 to 255 integer as the grayscale.
- Blue channel is density of the grid, or how namy points are there in this grid. In my case, each point contributes 25 in grayscale, so grid with 10 or more points will have 250 as grayscale.
Each grid is one pixel, there you have the RGB picture.

**Note that** you can for sure define your own channels if you find my defination not good enough. You may also define more than or less than 3 channels, but you have to dig into the codes since I coded assuming there are 3 channels, so sorry for that :P.

### 2.1.2 Size and resolution
I chose 304x304 as the input resolution, and with a 0.1m per grid I'm detecting in a area of 30.4mx30.4m, so 30.4m in front of the vehicle, and 15.2m each on left and right of it. I picked this resolution because:
- 0.1m is a fine compromise among accuracy, speed and calculation convenience.
- Kitti dataset only labels object inside camera angle, only about 120 degrees in the front, therefore most of the labeled objects are in this area.
- My convolutional layer has a stride of 16, so on a 304x304 input it gives a feature map of 19x19.

You can also define your own input resolution, here are some tips:
- Smaller grid size normally means better result, as well as more memory usage and computational cost.
- If you use Kitti or your own dataset where not all surrounding 360 degrees are object-labeled, do not project the unlabeled part or your network will be harvesting tons of false positives.
- Input resolution should be devided by 16.
- If you changed only the resolution without changing the grid size or channel defination or anything else, you don't have to re-train the network since it's a Fully Convolutional Network (FCN).

### 2.1.3 Format and path
As I mentioned before, I use `.png` as extension instead of `.jpg`. I would strongly recommand this to you because png pictures does not compress, so nothing would be messing with the grayscale of each channel that you designed.
Path-wise you should write the **absolute path** of each picture to either the training or the validating list (both .txt files), with each file taking up a line.

### 2.1.4 Ground truth
If you want to train the network you will need ground truths. Each ground truth takes one line, and a 5(7)-coordinate ground truth label comes in following order: class, x, y, w, l, rz, (z, h).
- **class**: class of object, in integer starting from 0.
- **x, y**: relative position of the center point of the rectangle, which are two floats range in (0, 1).
- **w, l**: relative size of the rectangle, which are also two floats range in (0, 1).
- **rz**: orientation, which range in (-pi/2, pi/2).
- **z**: relative height of the center of the cuboid, which is float range in (0, 1).
- **h**: relative height of the cuboid, which is float range in (0, 1).
For each picture there should be a **.txt** file with the same file name in ASCII format including all ground truth labels in it. E.g. for a picture `D:/0001.png` its ground truth may be found in `D:/0001.txt`.

**NOTE**: I did code another program to prepare Kitti dataset in the way I introduced, but it's not yet GitHub-ready. I'll put the link here when it is.

## 3 Configure the Network

There are two 'Cronus' networks in this project:
- **'Cronus'** detects vehicles in 2D plane, so it predicts a rectangle with orientation estimation. Therefore it outputs 5 coordinates.
- **'Cronus2'** also has two parts:
  - The **main** net detects vehicles with 2D data in 3D space, so it predicts a cuboid with orientation estimation. By assuming yaw and pitch to be 0, it outputs 7 coordinates.
  - The **sub** net, which does convolution to image input, takes main net output and convert to 2D box on image, takes 2D box as RoI then does RoI pooling on the feature map ,and runs a softmax classifier on the pooled feature.
  - Sub net is not yet implemented. RoI pooling layer took me too much time.

### 3.1 Data file
Data file is the fundament of configurating the network. It includes the training and validating list, as well as the configuration file.
Data file may be an ASCII file with any extension name, but in my case I used `.data`. So you may want to check out `cronus.data` or `cronus2.data` as an example.

### 3.2 Configuration file
Configuration file contains the structure of the network and property of each layer.
You may find the last layer as 'Region5' (for Cronus) or 'Region7' (for Cronus2), and it has some properties called 'num', 'coords' and 'classes'.
- 'Num' is how many anchor boxes there are. Anchor boxes are given in 'anchors' line, each has two float numbers describing its size. If 'bias_match' is set to 1 the network will automatically adapt its anchor according to ground truths, so you don't have to give 'anchors' value.
- 'Coords' is how many coordinates the networks outputs, so it should be 5 for Region5 (or for Cronus) and 7 for Region7 (or for Cronus2).
- 'Classes' is how many classes there are. In my case since I only detect vehicles in general it is 2.
And before Region5 or Region7, the 'filters' of the last convolutional layer should be `filters = num * (coords + classes + 1)`, where '+1' is for probability estimation. So in my case it has num = 3, coords = 7 and classes = 2, therefore filters = 3 * (7 + 2 + 1) = 30.

### 3.3 Training & validating list file
These two files are `.txt` files with training and validating sets, each picture in the set takes up a line.

## 4 Train your own network








## Under Construction

This code is for my thesis, where I modify and use Yolo on Velodyne data to detect surrounding vehicles on our autonomous driving platform

For now the network looks promising, on a 304x304 input in bird view, which is 30.4x30.4 meter, the network is able to predict 2D or even 3D boxes with orientation estimation within 20ms on a single GTX1080. With a IoU threshold of .6, the network managed to reach 70% recall (on KITTI dataset).

### Note that this repository is still under construction

and would remain this way before my thesis is finished. Any usage, including usage strictlly following my instruction, may lead to unexpected behavior.

This repository is forked from Yolo-Windows v2, but with a ton of midifications. I'll try explaining each of them as they are also part of my thesis.

Issue/direct contact is welcomed.
