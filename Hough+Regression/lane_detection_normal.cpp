#include <iostream>
#include <sstream>
#include<string>
#include <algorithm>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;



// median function
float median(vector<float> values) {
    sort(values.begin(), values.end());
    if (values.size() % 2 == 0) {
        return float((values[values.size()/2 - 1] + values[values.size()/2]) / 2.0);
    } else {
        return float(values[values.size()/2]);
    }
}

// standard deviation function
float standardDeviation(vector<float> values) {
    float meanv = 0.0, deviation = 0.0;
    int n = values.size();
    // calculate mean
   for (int i = 0; i < n; ++i) {
        meanv += values[i];
    }
    meanv /= n;
    // calculate deviation
    for (int i = 0; i < n; ++i) {
        deviation += (values[i] - meanv) * (values[i] - meanv);
        }
    deviation = float(sqrt(float(deviation / n)));
    return deviation;
}

// mean function
float mean(vector<float> values) {
     float sum = 0.0;
    int n = values.size();

    //  calculate sum
    for (int i = 0; i < n; ++i) {
         sum += values[i];
    }
    return float(sum / n);
    }

// median function
float median2(float values[], int n) {
    vector<float> v(values, values+n);
    sort(v.begin(), v.end());
    if (n % 2 == 0) {
        return float((v[n/2 - 1] + v[n/2]) / 2.0);
    } else {
        return float(v[n/2]);
    }
}

// standard deviation function
float standardDeviation2(float values[], int n) {
    float mean = 0.0, deviation = 0.0;
    // calculate mean
    for (int i = 0; i < n; ++i) {
        mean += values[i];
    }
    mean /= n;
    // calculate deviation
    for (int i = 0; i < n; ++i) {
        deviation += (values[i] - mean) * (values[i] - mean);
    }
    deviation = float(sqrt(float(deviation / n)));
    return deviation;
}




// Main function to run lane detection. 
int main() {
    // Load the dash cam video, throw error if video not found. 
    VideoCapture capture("/Users/anirudhramesh/Desktop/opencv/Self_Driving_Car/final_sub_code/drivingVideo.mp4");
    if (!capture.isOpened()) {
        cerr << "Error: Could not open the video file." << endl;
        return -1;
    }
   // Rect roi(425, 300, 150, 550);
    float x1Prev = 0 ;
    float x2Prev = 0;
    float y1Prev = 0;
    float y2Prev = 0;
    float y1NPrev = 0;
    float y2NPrev = 0;
    float x1NPrev = 0;
    float x2NPrev = 0;
    double fps = capture.get(CAP_PROP_FPS);
    Size size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    // Read in the frames and process them one by one.
    for (int i =0; i <215; i++){
       // Had this to skip alternate frames, then decided to run for all frames.
        if (i %2 == 0 or i%2 != 0){
            cout << "Frame: " << i << endl;
            Mat frame1, frame, frame2;
            capture.set(CAP_PROP_POS_FRAMES, i);
            capture >> frame;
            capture >> frame1;
            capture >> frame2;
            if (frame.empty()){
                cout << "Frame empty" << endl;
                break;
            }
            
            Size desired_output_size(800, 550);
            // Define the original four corners of the lane region
            vector<Point2f> src_pts = {
                Point2f(100, 600),
                Point2f(200, 375),
                Point2f(775, 375),
                Point2f(900, 600)
            };
            
            // Define the corresponding four corners in the bird's eye view
            vector<Point2f> dst_pts = {
                Point2f(0, 720),
                Point2f(0, 0),
                Point2f(720, 0),
                Point2f(720, 720)
            };

            // Compute the homography matrix
            Mat H = getPerspectiveTransform(src_pts, dst_pts);

            // Apply the transformation
            Mat transformed;
            warpPerspective(frame, transformed, H, desired_output_size);
            frame1 = frame; 
            
            // Image processing
            Mat gray, blurred, edges, roiMasked;
            cvtColor(frame1, gray, COLOR_BGR2GRAY);
            GaussianBlur(gray, blurred, Size(9, 9), 0);
            Canny(gray, edges, 50, 200);

            // Define the vertices of the polygon
            vector<Point> vertices;
            int x = frame1.rows;
            int y = frame1.cols; 
            vertices.push_back(Point(100,550));                 // Bottom-left corner
            vertices.push_back(Point(450, 320));                // Top-left corner
            vertices.push_back(Point(550, 320));                // Top-right corner
            vertices.push_back(Point(900,550));                 // Bottom-right corner

            // Create a mask for the polygon
            Mat mask = Mat::zeros(frame1.size(), CV_8UC1);
            fillConvexPoly(mask, vertices, Scalar(255));

            // Apply the mask to the input frame1
            edges.copyTo(roiMasked, mask);

            // DEFINE THE VECTORS TO STORE VALUES TO PLOT LATER 

            vector<vector<int>> slopePositiveLines; // x1 y1 x2 y2 slope
            vector<vector<int>> slopeNegativeLines;
            vector<float> slopespos1;
            vector<float> slopesneg1;
            vector<vector<int>> Lineinfo;
            vector<int> yValues;
            vector<float> positiveSlopes;
            vector<float> posSlopesGood;
            vector<float> negSlopesGood;
            vector<float> xInterceptPos;
            vector<float> xIntPosGood;
            vector<float> xInterceptNeg;
            vector<float> xIntNegGood;
            vector<vector<int>> slopePositiveLines2;
            vector<vector<int>> slopeNegativeLines2;


            // Define parameters and perform Hough Tranform to find lines from edges

            float rho = 2; // distance resolution in pixels of the Hough grid
            float theta = CV_PI/180; // angular resolution in radians of the Hough grid
            int threshold = 40; // minimum number of votes (intersections in Hough grid cell)
            int min_line_len = 40; // minimum number of pixels making up a line
            int max_line_gap =90; // maximum gap in pixels between connectable line segments
            vector<Vec4i> lines;
            HoughLinesP(roiMasked, lines, rho , theta, threshold, min_line_len, max_line_gap);
            cout << "Found "<<lines.size()<<" lines." << endl;

            // Plot Hough Lines
            // Check if lines are empty and plot 
            if (!lines.empty() && lines.size() > 2) {
                Mat allLines = frame1.clone();
                for (int i = 0; i < lines.size(); i++) {
                    Vec4i line2 = lines[i];
                    double x1 = line2[0];
                    double y1 = line2[1];
                    double x2 = line2[2];
                    double y2 = line2[3];
                    double slope = static_cast<float>((y2 - y1) / (x2 - x1));
                    double ang = abs(atan(slope) * 180 / CV_PI);
                    if (abs(ang) < 90 && abs(ang) > -90)
                        line(allLines, Point(line2[0], line2[1]), Point(line2[2], line2[3]), Scalar(255, 255, 0), 2); // plot line
                }   
                double alpha = 0.7; // weight of overlay

                // Loop through all the lines found using hough Transform and split into postivie and negative lanes 
                // depending on slope value. 
                bool addedPos = false;
                bool addedNeg = false;

                int count = 0;
                int count_neg = 0;
                for (size_t i = 0; i < lines.size(); i++) {
                    vector<Vec4i> currentLine;
                    currentLine.push_back(lines[i]);
                    int x1 = lines[i][0];
                    int y1 = lines[i][1];
                    int x2 = lines[i][2];
                    int y2 = lines[i][3];
                    float lineLength = sqrt(pow(x2-x1,2) + pow(y2-y1,2));        // Find Length
                    if (lineLength > 30) {                                  // If line is not too small, continue
                        if (x2 != x1) { 
                            float slope = float(y2 - y1) / float(x2 - x1);           // Find Slope 
                            // add to positive or negative slope depending on value
                            if (slope > 0){
                                float tanTheta =  tan(float(abs(y2 - y1)) / float(abs(x2 - x1))); // tan(theta) valuev
                                float ang = atan(tanTheta) * 180 / M_PI;  
                                if (abs(ang) < 90 && abs(ang) > 0) {
                                    slopePositiveLines.push_back({ x1, y1, x2, y2}); // add positive slope line
                                    slopespos1.push_back({-slope});
                                    count += 1;
                                    addedPos = true; // note that we added a positive slope line
                                }
                            }

                            if (slope < 0) {
                                float tanTheta =  tan(float(abs(y2 - y1)) / float(abs(x2 - x1))); // tan(theta) value
                                float ang = atan(tanTheta) * 180 / M_PI;
                                if (abs(ang) < 90 && abs(ang) > 0) {
                                    slopeNegativeLines.push_back({ x1, y1, x2, y2});    // add negative slope line
                                    slopesneg1.push_back({-slope});
                                    count_neg += 1;
                                    addedNeg = true; // note that we added a negative slope line
                                }
                            }
                        }
                    }
                }
            
                if (!addedPos) {
                    for (size_t i = 0; i < lines.size(); i++) {
                        Vec4i currentLine = lines[i];
                        int x1 = currentLine[0];
                        int y1 = currentLine[1];
                        int x2 = currentLine[2];
                        int y2 = currentLine[3];
                        float slope = float(y2 - y1) / float(x2 - x1);
            
                        if (slope > 0) {
                            // Check angle of line w/ xaxis. dont want vertical/horizontal lines
                            float tanTheta = tan(float(abs(y2 - y1)) / float(abs(x2 - x1)));
                            float ang = atan(tanTheta) * 180 / CV_PI;
                            if (abs(ang) < 90 && abs(ang) > 50) {
                                slopePositiveLines.push_back({ x1, y1, x2, y2}); // add negative slope line
                                slopespos1.push_back({-slope});
                                addedPos = true;
                            }
                        }
                    }
                }
                if (!addedNeg) {
                    for (size_t i = 0; i < lines.size(); i++) {
                        Vec4i currentLine = lines[i];
                        int x1 = currentLine[0], y1 = currentLine[1], x2 = currentLine[2], y2 = currentLine[3];
                        float slope = float(y2 - y1) / float(x2 - x1);
                        if (slope < 0) {
                            float tanTheta = tan(float(abs(y2 - y1)) / float(abs(x2 - x1))); // tan(theta) value
                            float ang = atan(tanTheta) * 180 / CV_PI;
                            if (abs(ang) < 90 && abs(ang) > 40) {
                                slopeNegativeLines.push_back({x1, y1, x2, y2});
                                slopesneg1.push_back({-slope});
                                addedNeg = true; 
                            }
                        }
                    }
                }    

                // Check if we have found any lines. 
                if (!addedPos || !addedNeg) {
                    cout << "Not enough lines found" << endl;
                }
        
                // ------------------------Get Positive/Negative Slope Averages-----------------------------------
                // Average position of lines and extrapolate to the top and bottom of the lane.
                for (float slope : slopespos1) {
                    positiveSlopes.push_back(slope);
                } 
       
                float posSlopeMedian = median(slopespos1);
                float posSlopeStdDev = standardDeviation(slopespos1);
                int countx = 0;
                for (float slope : positiveSlopes) {
                    // Check if 'x' values is close to the median, if not remove line from consideration for positive slope line
                    if (abs(slope - posSlopeMedian) <  0.2) {
                        posSlopesGood.push_back(slope);
                        countx = countx +  1;
                    }
                }
                // Check if 'x' values is close to the median, if not remove line from consideration for negative slope line
                float* negativeSlopes = new float[slopesneg1.size()];
                for(int i=0; i<slopesneg1.size(); i++){
                    negativeSlopes[i] = slopesneg1[i];
                }
                float negSlopeMedian = median2(negativeSlopes, slopesneg1.size());
                float negSlopeStdDev = standardDeviation2(negativeSlopes, slopesneg1.size());
                int count2 = 0;
                for(int i=0; i<slopeNegativeLines.size(); i++){
                    float slope = float(negativeSlopes[i]);
                if (abs(slope-negSlopeMedian) < 0.2){  
                    negSlopesGood.push_back(slope);
                    count2 = count2 + 1;
                    }
                }
                
                // --------------------------Get Average x Coord When y Coord Of Line = 0----------------------------
                // Positive Lines
        

                // Find xIntercept position for all postive lines. 
                for (int i = 0; i < slopePositiveLines.size(); i++) {
                    auto& line = slopePositiveLines[i];
                    float x1 = line[0];
                    float y1 = frame1.rows - line[1]; // y axis is flipped
                    float slope = slopespos1[i];
                    float yIntercept = y1 - slope * x1;
                    float xIntercept = float(-yIntercept / slope); // find x intercept based off y = mx+b
                    if (xIntercept == xIntercept) { // checks for nan
                        xInterceptPos.push_back(xIntercept); // add x intercept
                    }
                }
        
                // Find median of xIntercept
                float xIntPosMed = median(xInterceptPos);
                float xInterceptstd = standardDeviation(xInterceptPos);
                // If not near median we get rid of that x point
                for (int i = 0; i < slopePositiveLines.size(); i++) {
                    auto& line = slopePositiveLines[i];
                    float x1 = line[0];
                    float y1 = frame1.rows - line[1];
                    float slope = slopespos1[i];
                    float yIntercept = y1 - slope * x1;
                    float xIntercept = float(-yIntercept / slope);
                    if (abs(xIntercept - xIntPosMed) < xInterceptstd*1) { // check if near median
                        xIntPosGood.push_back(xIntercept);
                        slopePositiveLines2.push_back(line);
                    }
                }
                float xInterceptPosMean = mean(xIntPosGood);
                
                // Find xIntercept position for all negative lines. 
                for (int i = 0; i < slopeNegativeLines.size(); i++) {
                    auto& line = slopeNegativeLines[i];
                    float x1 = line[0];
                    float y1 = frame1.rows - line[1];
                    float slope = slopesneg1[i];
                    float yIntercept = y1 - slope * x1;
                    float xIntercept = float(-yIntercept / slope);
                    if (xIntercept == xIntercept) { // check for nan
                    xInterceptNeg.push_back(xIntercept);
                    }
                }
                // Find median of xIntercept
                float xIntNegMed = mean(xInterceptNeg);
                float xInterceptstdN = standardDeviation(xInterceptNeg);
        
        
                for (int i = 0; i < slopeNegativeLines.size(); i++) {
                    auto& line = slopeNegativeLines[i];
                    float x1 = line[0];
                    float y1 = frame1.rows - line[1];
                    float slope = slopesneg1[i];
                    float yIntercept = y1 - slope * x1;
                    float xIntercept = float(-yIntercept / slope);
                    if (abs(xIntercept - xIntNegMed) <xInterceptstdN*1) {   
                        xIntNegGood.push_back(xIntercept);
                        slopeNegativeLines2.push_back(line);
                    }
                }
                float xInterceptNegMean = mean(xInterceptNeg);
            }


            // ------------------------Perform Linear Regression and plot the line-----------------------------------
            bool addedP = false;
            bool addedN = false;
            Mat points(slopePositiveLines2.size(), 2, CV_32F);
            for (int i = 0; i < slopePositiveLines2.size(); i++) {
                if (slopePositiveLines2[i][1] != 0) {
                    points.at<float>(i, 0) = slopePositiveLines2[i][0];  // x coordinate
                    points.at<float>(i, 1) = slopePositiveLines2[i][1];  // y coordinate
                    addedP = true;
                }
            }
            Mat pointsN(slopeNegativeLines2.size(), 2, CV_32F);
            for (int i = 0; i < slopeNegativeLines2.size(); i++) {
                if (slopeNegativeLines2[i][1] != 0) {
                    pointsN.at<float>(i, 0) = slopeNegativeLines2[i][0];  // x coordinate
                    pointsN.at<float>(i, 1) = slopeNegativeLines2[i][1];  // y coordinate
                    addedN = true;
                }
            }
            Vec4f FitedLine, FitedLineN;;
            // If there are no positive lines or no negative lines we can't do anything, else fit a line 
            if (addedP && addedN) {

                fitLine(points, FitedLine, DIST_L2, 0, 0.01, 0.01);
                int t0 = (350-FitedLine[3])/FitedLine[1]; // Calculate the x-coordinate for y = 350
                int t1 = (frame1.rows -FitedLine[3])/FitedLine[1]; // Calculate the x-coordinate for y = frame.rows
                Point p0 = Point(FitedLine[2], FitedLine[3]) + Point(t0 * FitedLine[0], t0 * FitedLine[1]); // calculate point on the line using the x,y cords and to, t1
                Point p1 = Point(FitedLine[2], FitedLine[3]) + Point(t1 * FitedLine[0], (t1 * FitedLine[1])); // calculate point on the line using the x,y cords and to, t1


                int x1 = round(p0.x);
                int x2 = round(p1.x);
                int y1 = round(p0.y);
                int y2 = round(p1.y);
                int jumpThresh = 80;
                int jumpThresh2 = 3;

                // Check with value for previous frame change in the direction if the x-cord changes by more than 2 pixels. 
                if (x1Prev != 0 && x2Prev != 0){ // check if we have a previous point
                    if (abs(x1 - x1Prev) > 2 && abs(x1 - x1Prev) < jumpThresh) {     // if x pos differs from xprev too much just get the direction and move it a small amount
                        if (x1 - x1Prev > 0){
                            p0.x = x1Prev + 1;
                        }
                        else {
                            p0.x = x1Prev -1;
                        }
                    }
                    else if (abs(x1-x1Prev) >= jumpThresh){
                        p0.x = x1Prev;
                    }
                    if (abs(y1 - y1Prev) > 3 && abs(y1 - y1Prev) < jumpThresh2) {     // if x pos differs from xprev too much just get the direction and move it a small amount
                        if (y1 - y1Prev > 0){
                            p0.y = y1Prev + 1;
                        }
                        else {
                            p0.y = y1Prev -1;
                        }
                    }
                    else if (abs(y1-y1Prev) >= jumpThresh2){
                        p0.y = y1Prev;
                    }

                    if (abs(x2 - x2Prev) > 2 && abs(x2 - x2Prev) < jumpThresh) { // if x pos differs a lot, maybe we had the wrong line, so jumping is allowed then
                        if (x2 - x2Prev > 0){
                            p1.x = int(x2Prev + 1);
                        }
                        else {
                            p1.x = int(x2Prev -1);
                        }
                    }
                    else if (abs(x2-x2Prev) >= jumpThresh){
                        p1.x = x2Prev;
                    }
                    if (abs(y2 - y2Prev) > 1 && abs(y2 - y2Prev) < jumpThresh2) {     // if x pos differs from xprev too much just get the direction and move it a small amount
                        if (y2 - y2Prev > 0){
                            p1.y = y2Prev + 1;
                        }
                        else {
                            p1.y = y2Prev -1;
                        }
                    }
                    else if (abs(y2-y2Prev) >= jumpThresh2){
                        p1.y = y2Prev;
                    }
                    if ((isnan(p1.x))) {
                        p1.x = x2Prev;    
                    }
                    if ((isnan(p1.y))) {
                        p1.y = y2Prev;    
                    }
                    if ((isnan(p0.x))) {
                        p1.x = x1Prev;    
                    }
                    if ((isnan(p0.y))) {
                        p0.y = y1Prev;    
                    }

                }

                // Same process as above to get negative line 
                fitLine(pointsN, FitedLineN, DIST_L2, 0, 0.01, 0.01);            
                int t0N = (350-FitedLineN[3])/FitedLineN[1];
                int t1N = (frame1.rows -FitedLineN[3])/FitedLineN[1];
                Point p0N = Point(FitedLineN[2], FitedLineN[3]) + Point(t0N * FitedLineN[0], t0N * FitedLineN[1]);
                Point p1N = Point(FitedLineN[2], FitedLineN[3]) + Point(t1N * FitedLineN[0], t1N * FitedLineN[1]);

            
                int x1N = round(p0N.x);
                int x2N = round(p1N.x);
                int y1N = round(p0N.y);
                int y2N = round(p1N.y);
                if (x1NPrev != 0) {
                    if (abs(x1N-x1NPrev) > 2 && abs(x1N-x1NPrev) < jumpThresh) {
                        if (x1N - x1NPrev > 0){
                            p0N.x = int(x1NPrev +1);
                        }
                        else {
                            p0N.x = int(x1NPrev - 1);
                        }
                    }
                    else if (abs(x1N-x1NPrev) >= jumpThresh){
                        p0N.x = x1NPrev;
                    }
                    if (abs(y1N  - y1NPrev) > 3 && abs(y1N - y1NPrev) < jumpThresh2) {     // if x pos differs from xprev too much just get the direction and move it a small amount
                        if (y1N - y1NPrev > 0){
                            p0N.y = y1NPrev + 1;
                        }
                        else {
                            p0N.y = y1NPrev -1;
                        }
                    }
                    else if (abs(y1N-y1NPrev) >= jumpThresh2){
                        p0N.y = y1NPrev;
                    }
    
                    if (abs(x2N-x2NPrev) > 2 && abs(x2N-x2NPrev) < jumpThresh) {
                        if (x2N - x2NPrev > 0){
                            p1N.x = int(x2NPrev +1);
                        }
                        else {
                            p1N.x = int(x2NPrev - 1);
                        }
                    }
                    else if (abs(x2N-x2NPrev) >= jumpThresh){
                        p1N.x = x2NPrev;
                    }
                    if (abs(y2N - y2NPrev) > 1 && abs(y2N - y2NPrev) < jumpThresh2) {     // if x pos differs from xprev too much just get the direction and move it a small amount
                        if (y2N - y2NPrev > 0){
                            p1N.y = y2NPrev + 1;
                        }
                        else {
                            p1N.y = y2NPrev -1;
                        }
                    }
                    else if (abs(y2N-y2NPrev) >= jumpThresh2){
                        p1N.y = y2NPrev;
                    }

                    if ((isnan(p0N.x ))) {
                        p0N.x = x1NPrev;    
                    }
                    if ((isnan(p1N.x ))) {
                        p1N.x = x2NPrev;    
                    }
                    if ((isnan(p0N.y))) {
                        p0N.y = y1NPrev;    
                    }
                    if ((isnan(p1N.y))) {
                        p1N.y = y2NPrev;    
                    }
                } 


                // Plot the line 
                line(frame1, p0, p1, Scalar(0, 255, 0), 3);
                line(frame1, p0N, p1N, Scalar(0, 255, 0), 3);
                cout << p0 << ":PO      P1:" << p1 << endl;
                cout << p0N << ":PoN    P1N:" << p1N <<endl;
                

                // Store values to use with next frame
                if (x1 != 0) { 
                    x1Prev = p0.x;
                }
                if (x2 != 0) { 
                    x2Prev = p1.x;
                }
                if (x1N != 0) { 
                x1NPrev = p0N.x;
                }
                if (x2N != 0) { 
                x2NPrev = p1N.x;
                }

                if (y1 != 0) { 
                    y1Prev = p0.y;
                }
                if (y2 != 0) { 
                    y2Prev = p1.y;
                }
                if (y1N != 0) { 
                y1NPrev = p0N.y;
                }
                if (y2N != 0) { 
                y2NPrev = p1N.y;
                }
                // Take the 4 points used to draw the lines and create a mask and overaly and highlight the lane. 

                vector<Point> points2; // create a vector to store the points of the polygon to be filled
                // add the points of the polygon to the vector
                points2.push_back( Point(p1N.x,p1N.y));
                points2.push_back(Point(p1.x,  p1.y));
                points2.push_back(Point(p0.x,  p0.y));
                points2.push_back(Point(p0N.x,p0N.y));
                            
                // create an array of polygons
                vector<vector<Point>> polygons;
                polygons.push_back(points2);

                // create a mask image of the same size as the color image

                Mat mask2 = Mat::zeros(frame1.size(), CV_8UC1);
                Mat Blended_image;
                // fill the polygon with the specified color in the mask image
                double alpha = 0.5; // weight of overlay
                cvtColor(mask2, mask2, COLOR_GRAY2BGR);
                fillPoly(mask2, polygons, Scalar(255,0,0));
                addWeighted(frame1, 1, mask2, 1.0 - alpha, 0, Blended_image); 

                // show final output 
                imshow("final", Blended_image);
                if (waitKey(30) >= 27)
                break;  


            }
        }
    }
}



