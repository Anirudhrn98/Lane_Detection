# Lane Detection Using C++

This repository contains the implementation details of a lane detection system for autonomous driving. The repository contains separate folders containing the code for different pipelines used to detect the lanes. 

These pipelines include:

1. Using hough lines and linear regression.
2. Using hough lines and linear regression along with using the information from previous frames.
3. Using Homography to convert input frame to bird's eye point of view.

# Packages required to run the code 

You need to have opencv installed on your system to run the code. 

# How to run the code

For each of the folders i.e. "Hough+Regression", "Hough+Regression+Learn", "Homography", steps are the same, they are as follows, 

1. Open CmakeLists.txt and replace "path_to_opencv" with the path to where you have opencv installed in your system. 
2. Access the folder using your terminal.
3. Once you are in the folder, use the command
   ```bash
   cd build
   ```
   This will create the build directory for your program.
4. Run this command to compile 
   ```bash
   make
   ```
5. Run this command to run the code (replace file_name, with the name of the cpp file you want to run), make to replace the path to your input video file before running the code.
   ```bash
   ./file_name
   ```



