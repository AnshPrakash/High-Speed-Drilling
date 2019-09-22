#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <bits/stdc++.h>

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

cv::Mat transform(const cv::Mat& image){

}

int main(int argc, char** argv){
  cv::Mat colImage = cv::imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
  std::cout<<type2str(image.type())<<" : gray image \n";
  std::cout<<type2str(colImage.type())<<" : coloured image \n";


  cv::namedWindow("original",CV_WINDOW_FREERATIO);
  cv::imshow("original",image);

  
  cv::namedWindow( "Perspective Transform", CV_WINDOW_FREERATIO );
  cv::imshow( "Perspective Transform", transformed );
  

  
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}