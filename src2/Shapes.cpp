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

cv::Mat getEdges(const cv::Mat& image){
  //used only for houghman line
  int edgeThresh = 1;
  int lowThreshold = 50;
  // int const max_lowThreshold = 100;
  int ratio = 2;
  int kernel_size = 3;
  cv::Mat detectedEdges;
  int d = 9; //d is filter size
  cv::bilateralFilter(image,detectedEdges,d,157,157);
  cv::medianBlur(detectedEdges,detectedEdges,9);
  // cv::namedWindow("Image",CV_WINDOW_FREERATIO);
  // cv::imshow("Image",detectedEdges);
  // cv::waitKey(0);
  // cv::destroyAllWindows();
  cv::Canny( detectedEdges, detectedEdges, lowThreshold, lowThreshold*ratio, kernel_size );
  return(detectedEdges);
}

cv::Mat getCircles(const cv::Mat& image){
  int d = 9;
  cv::Mat edges;
  // std::cout<<type2str(edges.type())<<"\n";
  cv::bilateralFilter(image,edges,d,157,157);
  std::vector<cv::Vec3f> circles;
  double minDist = edges.rows/2.3;
  int const max_lowThreshold = 50;
  int ther_votes = 15;
  int min_radius = 45;
  int max_radius = 70;
  int dp = 1;
  cv::HoughCircles( edges, circles, CV_HOUGH_GRADIENT,dp, minDist, max_lowThreshold, ther_votes, min_radius, max_radius );
  std::cout<<circles.size()<<" :No. of Circles detetected\n";
  // Draw the circles detected
  for( size_t i = 0; i < circles.size(); i++ ){
    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    // circle center
    circle( edges, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
    // circle outline
    std::cout<<radius<<"\n";
    circle( edges, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );
   }
  return(edges);
}

cv::Mat getLines(const cv::Mat& edges,const cv::Mat& image){
  // cv::Mat 
  cv::Mat display = image.clone();
  std::vector<cv::Vec2f> lines;
  double rho,theta,srn,stn,min_theta,max_theta;
  min_theta = 0;
  rho = 1;theta = CV_PI/180;
  max_theta = CV_PI/1.7;
  int thershold = 40;
  srn = 0.0; stn = 0.0;
  cv::HoughLines(edges,lines,rho,theta,thershold,srn,stn,min_theta,max_theta);
  std::cout<<lines.size()<<" :Lines \n";
  for( size_t i = 0; i < lines.size(); i++ ){
    float rho = lines[i][0], theta = lines[i][1];
    cv::Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
    cv::line( display, pt1, pt2, cv::Scalar(0,34,255),1, cv::LINE_AA);
  }
  return(display);
}

int main(int argc, char** argv){
  cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
  std::cout<<type2str(image.type())<<" Before\n";
  
  cv::Mat edges = getEdges(image);
  // edges.convertTo(edges,CV_8UC1);
  cv::Mat circled = getCircles(image);
  cv::Mat lines = getLines(edges,image);

  cv::namedWindow("original",CV_WINDOW_FREERATIO);
  cv::imshow("original",image);


  cv::namedWindow("Edges",CV_WINDOW_FREERATIO);
  cv::imshow("Edges",edges);

  cv::namedWindow( "Hough Circle Transform", CV_WINDOW_FREERATIO );
  cv::imshow( "Hough Circle Transform", circled );

  cv::namedWindow( "Hough Line Transform", CV_WINDOW_FREERATIO );
  cv::imshow( "Hough Line Transform", lines );
  

  
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}