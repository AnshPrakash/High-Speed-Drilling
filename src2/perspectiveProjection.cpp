#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <bits/stdc++.h>

cv::Mat colImage;

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

int No_Of_Point;

void CallBackFunc(int event, int x, int y, int flags, void* ptr){
  std::vector<cv::Point2f> *vec = static_cast<std::vector<cv::Point2f> *>(ptr);
  if( event == cv::EVENT_LBUTTONDOWN ){
    if(No_Of_Point < 4) {
      vec->push_back(cv::Point2f(x,y));
      // circle(colImage,cv::Point2f(x,y),4,cv::Scalar(0,0,255),2, 8, 0);
      No_Of_Point++;
      std::cout<<(*vec)<<"\n";
    }
    std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" <<"\n";
  }
  else if  ( event == cv::EVENT_RBUTTONDOWN ){

    std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << "\n";
  }
  else if  ( event == cv::EVENT_MBUTTONDOWN ){
    std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << "\n";
  }
  
}

int main(int argc, char** argv){
  No_Of_Point = 0;
  colImage = cv::imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
  std::cout<<type2str(image.type())<<" : gray image \n";
  std::cout<<type2str(colImage.type())<<" : coloured image \n";

  // image Quadilateral or Image plane coordinates
  cv::Point2f imageQuad[4]; 
  // Output Quadilateral or World plane coordinates
  cv::Point2f outputQuad[4];
        
  // Lambda Matrix
  cv::Mat lambda( 2, 4, CV_32FC1 );
  //image and Output Image;
  cv::Mat output;
  
  // Set the lambda matrix the same type and size as image
  lambda = cv::Mat::zeros( image.rows, image.cols, image.type() );

  std::vector<cv::Point2f> vec;
  cv::namedWindow("My Window", CV_WINDOW_FREERATIO);
  cv::setMouseCallback("My Window", CallBackFunc,&vec);
  cv::imshow("My Window",colImage);
  cv::waitKey(0);
  cv::destroyAllWindows();
  // The 4 points that select quadilateral on the image , from top-left in clockwise order
  // These four pts are the sides of the rect box used as image 
  for(int i = 0; i < vec.size(); i++ ) imageQuad[i] = vec[i];
  if(vec.size()<4 ) return(-1);
  // imageQuad[0] = cv::Point2f( 30,60 );
  // imageQuad[1] = cv::Point2f( image.cols-50,-50);
  // imageQuad[2] = cv::Point2f( image.cols-100,image.rows-50);
  // imageQuad[3] = cv::Point2f( 50,image.rows - 50  );  
  
  for(int i = 0; i < 4; i++ ){
    cv::Point center(cvRound(imageQuad[i].x), cvRound(imageQuad[i].y));
    // int radius = cvRound(circles[i][2]);
    circle( colImage, center, 3, cv::Scalar(0,255,0),2, 8, 0 );
  }
  // The 4 points where the mapping is to be done , from top-left in clockwise order
  outputQuad[0] = cv::Point2f( 0,0 );
  outputQuad[1] = cv::Point2f( image.cols-1,50);
  outputQuad[2] = cv::Point2f( image.cols-1,image.rows-1);
  outputQuad[3] = cv::Point2f( 0,image.rows-1  );

  // Get the Perspective Transform Matrix i.e. lambda 
  lambda = cv::getPerspectiveTransform( imageQuad, outputQuad );
  // Apply the Perspective Transform just found to the src image
  cv::warpPerspective(image,output,lambda,output.size() );


  //Create a window

  //set the callback function for any mouse event
  // cv::namedWindow("original",CV_WINDOW_FREERATIO);
  cv::imshow("original",colImage);
  cv::imshow("Output",output);

  
  // cv::namedWindow( "Perspective Transform", CV_WINDOW_FREERATIO );
  // cv::imshow( "Perspective Transform", transformed );
  

  
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}