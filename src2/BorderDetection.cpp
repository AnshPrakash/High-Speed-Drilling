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
  int lowThreshold = 45;
  int ratio = 2;
  int kernel_size = 3;
  cv::Mat detectedEdges;
  int d = 9; //d is filter size
  cv::bilateralFilter(image,detectedEdges,d,157,157);
  cv::medianBlur(detectedEdges,detectedEdges,9);
  cv::Canny( detectedEdges, detectedEdges, lowThreshold, lowThreshold*ratio, kernel_size );
  return(detectedEdges);
}


std::vector<cv::Vec2f> getLines(const cv::Mat& edges,const cv::Mat& image){
  cv::Mat display = image.clone();
  std::vector<cv::Vec2f> lines;
  double rho,theta,srn,stn,min_theta,max_theta;
  min_theta = 0;
  rho = 1;theta = CV_PI/180;
  max_theta = CV_PI;
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
  cv::namedWindow( "Hough Line Transform", CV_WINDOW_FREERATIO );
  cv::imshow( "Hough Line Transform", display );
  cv::waitKey(0);
  cv::destroyAllWindows();
  return(lines);
}

cv::Mat clusteringLines(std::vector<cv::Vec2f> lines, const cv::Mat& image){
  cv::Mat display = image.clone();
  cv::Mat labels;
  //normalise rhos as it can be too large and will give weightage to theta in k-mean clustering
  for( size_t i = 0; i < lines.size(); i++ ){
    lines[i][0] = 100.0*(lines[i][0])/(std::max(image.rows,image.cols));
    lines[i][1] = 100.0*(lines[i][1])/CV_PI;
  }
  int attempts = 3;
  cv::Mat centers;
  cv::kmeans(lines,4,labels,cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,600,0.0001),attempts,cv::KMEANS_PP_CENTERS,centers);
  std::cout<<type2str(labels.type())<<" : labels  \n";
  for( size_t i = 0; i < centers.rows; i++ ){
    float rho = centers.at<float>(i,0)*(std::max(image.rows,image.cols))/100.0;
    float theta = centers.at<float>(i,1)*CV_PI/100.0;
    cv::Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
    cv::line( display, pt1, pt2, cv::Scalar(0,34,255),1, cv::LINE_AA);
  }
  for( size_t i = 0; i < lines.size(); i++ ){
    float rho = lines[i][0]*(std::max(image.rows,image.cols))/100.0;
    float theta = lines[i][1]*CV_PI/100.0;
    cv::Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
    if(labels.at<int>(i) == 0) cv::line( display, pt1, pt2, cv::Scalar(13,255,5),1, cv::LINE_AA);
    if(labels.at<int>(i) == 1) cv::line( display, pt1, pt2, cv::Scalar(114,13,225),1, cv::LINE_AA);
    if(labels.at<int>(i) == 2) cv::line( display, pt1, pt2, cv::Scalar(252,0,55),1, cv::LINE_AA);
    if(labels.at<int>(i) == 3) cv::line( display, pt1, pt2, cv::Scalar(35,134,255),1, cv::LINE_AA);
  }
  return(display);
}

int main(int argc, char** argv){
  cv::Mat colImage = cv::imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
  std::cout<<type2str(image.type())<<" Before\n";
  
  cv::Mat edges = getEdges(image);
  // cv::Mat lines = getLines(edges,image);
  std::vector<cv::Vec2f> lines = getLines(edges,image);
  cv::Mat cen_line = clusteringLines(lines,colImage);
  // std::cout<<clusteringLines(lines,image)<<"\n";


  cv::namedWindow("original",CV_WINDOW_FREERATIO);
  cv::imshow("original",image);

  cv::namedWindow("Edges",CV_WINDOW_FREERATIO);
  cv::imshow("Edges",edges);

  cv::namedWindow( "Hough Line Transform", CV_WINDOW_FREERATIO );
  cv::imshow( "Hough Line Transform", cen_line );
  

  
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}