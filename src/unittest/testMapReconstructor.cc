
#include "KeyFrame_stub.h"
#include<iostream>
#include<fstream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace ORB_SLAM2;

 struct RcKeyPoint
    {
        cv::Point2f pt;
     

        // depth from mesurements
        float mDepth;


      

        /**
       * Constructor
       */
        RcKeyPoint(){}

        RcKeyPoint(float x, float y,float depth):
            pt(x,y),mDepth(depth){}

        // with depth sensor
        void setMDepth(float depth){
            mDepth = depth;
        }

     
    };

bool getSearchAreaForWorld3DPointInKF ( KeyFrameStub* const  pKF1, KeyFrameStub* const pKF2, const RcKeyPoint& twoDPoint,int& lowerBoundXInKF2, int& lowerBoundYInKF2, int& upperBoundXInKF2, int& upperBoundYInKF2 );

int main(int argc, char **argv)
{

    KeyFrameStub KF1;
    KeyFrameStub KF2;
   //    float N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    RcKeyPoint twoDPoint(110.0, 330.1, 3.25);

    cv::Mat pose1 = (cv::Mat_<float>(4,4) <<0.99300158, 0.0086956564, 0.11778051, 0.02448,
                                                                          0.011409527, 0.9855575, -0.16895621, -0.080968067,
                                                                          -0.11754865, 0.1691176, 0.97856098, -0.19543599,
                                                                          0, 0, 0, 1);
  cv::Mat pose2 = (cv::Mat_<float>(4,4) <<0.99273348, 0.011204797, 0.11981117, 0.02331865,
                                                                         0.0096331481, 0.98506004, -0.17194179, -0.078809343,
                                                                         -0.11994776, 0.17184652, 0.97779411, -0.19595753,
                                                                         0, 0, 0, 1);
    KF1.SetPose(pose1);
    KF2.SetPose(pose2);
    int lowerX, lowerY, upperX, upperY;
    bool validResult;
    validResult = getSearchAreaForWorld3DPointInKF (&KF1, &KF2,  twoDPoint,lowerX, lowerY, upperX, upperY );   
    
    

cout << lowerX << " "<<upperX << " "<< lowerY <<" "<< upperY<<endl;

    return 0;
}


bool getSearchAreaForWorld3DPointInKF ( KeyFrameStub* const  pKF1, KeyFrameStub* const pKF2, const RcKeyPoint& twoDPoint,int& lowerBoundXInKF2, int& lowerBoundYInKF2, int& upperBoundXInKF2, int& upperBoundYInKF2 )
{
    //Uproject lower and upper point from KF1 to world
    const float z = twoDPoint.mDepth;
    const float xDelta = 0.01*z, yDelta = 0.01*z;
    cv::Mat P3DcEst = cv::Mat();
    cv::Mat lower3Dw = cv::Mat();
     cv::Mat upper3Dw = cv::Mat();
      cv::Mat KF1Twc = cv::Mat();
     vector<cv::Mat>  boundPoints;
     boundPoints.reserve(8);  //todo: to configurable, considering the deviation in (R,t), in depth so it has 3d distribution cubic.
    if(z>1 && z<8)  //todo: to configurable, depth <= 1m is not good for RGBD sensor, depth >=8 m cause the depth distribution not sensitive.
    {
        float ZcBound[] = {0.95*z, 1.05*z};  //todo: to configurable, assume the depth estimation has 5% portion of accuracy deviation
   
        const float u = twoDPoint.pt.x;
        const float v = twoDPoint.pt.y;
  
       P3DcEst  = pKF1->UnprojectToCameraCoord(u,v,z);
       KF1Twc = pKF1->GetPoseInverse();
       
        float XcEst = P3DcEst.at<float>(0);
        float YcEst = P3DcEst.at<float>(1);
        cout <<"Xc estimation: "<< XcEst << " Yc estimation:"<<YcEst << "  Zc estimation"<< z<<endl;
        float XcBound[]= {XcEst-xDelta, XcEst+xDelta};
        float YcBound[]= { YcEst-yDelta,  YcEst+yDelta};
       for ( int xindex = 0; xindex < 2; xindex ++)
       {
           cv::Mat P3Dc = (cv::Mat_<float>(3,1) <<XcBound[xindex], YcEst, z);
                      //  cout<<xindex<<"," <<yindex<<"," <<zindex<<"{" <<"Xc bound: "<< XcBound[xindex] << " Yc bound:"<<YcBound[yindex] << "  Zc bound"<< ZcBound[zindex]<<endl;
                    cv::Mat P3Dw=KF1Twc.rowRange(0,3).colRange(0,3)*P3Dc+KF1Twc.rowRange(0,3).col(3);
               cout<<"Xw bound: "<< P3Dw.at<float>(0) << " Yw bound:"<<P3Dw.at<float>(1)  << "  Zw bound"<< P3Dw.at<float>(2)  <<"}"<<endl;
                 
                    boundPoints.push_back( P3Dw);
       }
       for ( int yindex = 0; yindex < 2; yindex ++)
       {
           cv::Mat P3Dc = (cv::Mat_<float>(3,1) <<XcEst, YcBound[yindex], z);
                      //  cout<<xindex<<"," <<yindex<<"," <<zindex<<"{" <<"Xc bound: "<< XcBound[xindex] << " Yc bound:"<<YcBound[yindex] << "  Zc bound"<< ZcBound[zindex]<<endl;
                    cv::Mat P3Dw=KF1Twc.rowRange(0,3).colRange(0,3)*P3Dc+KF1Twc.rowRange(0,3).col(3);
               cout<<"Xw bound: "<< P3Dw.at<float>(0) << " Yw bound:"<<P3Dw.at<float>(1)  << "  Zw bound"<< P3Dw.at<float>(2)  <<"}"<<endl;
                 
                    boundPoints.push_back( P3Dw);
       }
       for ( int zindex = 0; zindex < 2; zindex ++)
       {
                cv::Mat P3Dc = (cv::Mat_<float>(3,1) <<XcEst, YcEst, ZcBound[zindex]);
                       // cout<<xindex<<"," <<yindex<<"," <<zindex<<"{" <<"Xc bound: "<< XcBound[xindex] << " Yc bound:"<<YcBound[yindex] << "  Zc bound"<< ZcBound[zindex]<<endl;
                    cv::Mat P3Dw=KF1Twc.rowRange(0,3).colRange(0,3)*P3Dc+KF1Twc.rowRange(0,3).col(3);
               cout<<"Xw bound: "<< P3Dw.at<float>(0) << " Yw bound:"<<P3Dw.at<float>(1)  << "  Zw bound"<< P3Dw.at<float>(2)  <<"}"<<endl;
                 
                    boundPoints.push_back( P3Dw);
       }
           
// --------------------------------------------------------------------------
//to compare adjust in world coordination
//--------------------------------------------------------------------------      


          
    }
    else
        return false;
    
    //Project to  Neighbor KeyFrames
    
    float upperU = 0.0;
    float upperV =0.0;
    float lowerU = 0.0;
    float lowerV = 0.0;
    float tempU = 0.0;
    float tempV = 0.0;
    bool valid = false;
   bool firstround = true;
    
    //currently only roughly search the area to ensure all deviation are covered in the search area.
    cout <<"bound points"<<endl;
    for(auto & bp: boundPoints)
    {
         valid = pKF2->ProjectStereo(bp, tempU, tempV);
         if(!valid) 
                return false;
        
         cout << tempU<< "  " <<       tempV << endl;
         if(firstround)
         {
             firstround = false;
             upperU = lowerU = tempU;
             upperV = lowerV = tempV;
             continue;
         }
        if ( tempU > upperU)
            upperU = tempU;
        
        if (tempU < lowerU)
            lowerU = tempU;
            
        if(tempV > upperV)
            upperV = tempV;
            
        if(tempV< lowerV)
            lowerV = tempV;
    }


lowerBoundXInKF2 = floor(lowerU);
lowerBoundYInKF2 = floor(lowerV);
upperBoundXInKF2 = ceil(upperU);
upperBoundYInKF2 = ceil(upperV);
   
    return true;
    
}


