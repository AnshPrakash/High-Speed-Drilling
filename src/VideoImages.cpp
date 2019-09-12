
// This program is to extract images from the Medical Videos
#include <bits/stdc++.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <sys/stat.h> 
#include <sys/types.h> 


/*
 *
 *  f : fastforward
 *  n : normalframerate
 *  c : save this frame
 *  p : pause
 * 
 */

std::string type2str(int type) {
  std::string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  r += "C";
  r += (chans+'0');
  return r;
}


int main(int argc, char const *argv[]){
    /*
     *
     *  argv[1] should contain the name of the new folder to be created
     *  argv[2] should contain the name of video
     * 
     */

    std::string parentFolder = "../MedicalImages";
    if (mkdir("../MedicalImages", 0777) == -1) {
        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
    }
    else std::cout << "Empty Directory created\n";
    std::string newEntry = parentFolder +"/" + argv[1];
    char cstr[newEntry.size() + 1];

    newEntry.copy(cstr,newEntry.size() + 1);
    cstr[newEntry.size()] ='\0';
    if (mkdir(cstr, 0777) == -1) {
        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
    }
    else std::cout << "New Data directory created \n";
    

    cv::VideoCapture cap(argv[2]); 
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file\n";
        return -1;
    }
    
    int imageCount = 0;
    int displayTime = 25;
    int skipFrames = 200;
    bool fastForward = false;
    bool pause = false;
    cv::Mat frame;
    while(1){
        if (!pause) cap >> frame;
        if(!pause && fastForward){
            while (skipFrames >= 0){
                cap >> frame;
                if (frame.empty()) break;
                skipFrames--;
            }
            skipFrames = 100;
        }
        if (frame.empty()) break;
        cv::imshow( "Frames", frame );
        char c = (char)cv::waitKey(displayTime);
        if(c==27) break;
        if(c == 'c') cv::imwrite(newEntry +"/"+ std::to_string(++imageCount) +".jpg",frame);
        if(c == 'f') fastForward = true;
        if(c == 'n') fastForward = false;
        if(c == 'p') pause = !pause;
    }
    // When everything done, release the video capture object
    cap.release();
    cv::destroyAllWindows();
        
    return 0;
}
