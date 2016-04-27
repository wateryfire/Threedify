
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

    RcKeyPoint twoDPoint(111.0, 230.1, 7.25);

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
    cv::Mat lower3Dw = cv::Mat();
     cv::Mat upper3Dw = cv::Mat();
     vector<cv::Mat>  boundPoints;
     boundPoints.reserve(8);  //todo: to configurable, considering the deviation in (R,t), in depth so it has 3d distribution cubic.
    if(z>1 && z<8)  //todo: to configurable, depth <= 1m is not good for RGBD sensor, depth >=8 m cause the depth distribution not sensitive.
    {
        float lowerZ = 0.95*z;  //todo: to configurable, assume the depth estimation has 5% portion of accuracy deviation
        float upperZ = 1.05*z;  //todo: to configurable, assume the depth estimation has 5% portion of accuracy deviation
        const float u = twoDPoint.pt.x;
        const float v = twoDPoint.pt.y;
  
       lower3Dw = pKF1->UnprojectStereo(u,v, lowerZ);
       float x3D1=lower3Dw.at<float>(0);
       float y3D1=lower3Dw.at<float>(1);
       float bound[]= {0.98*x3D1, 1.02*x3D1};
       vector<float> xBound1(bound, bound+2);//todo: to configurable, assume the deviation in (R,t) estimation has 2% portion of accuracy deviation.
       bound= {0.98*y3D1, 1.02*y3D1};
       vector<float> yBound1 (bound, bound+2);//todo: to configurable, assume the deviation in (R,t) estimation has 2% portion of accuracy deviation.
       for(auto & x: xBound1)
       {
           for(auto & y: yBound1)
               {
                    boundPoints.push_back((cv::Mat_<float>(3,1) << x, y, lowerZ));
                }
        }
       
       upper3Dw = pKF1->UnprojectStereo(u,v, upperZ);
        
        float x3D2=upper3Dw.at<float>(0);
        float y3D2=upper3Dw.at<float>(1);
        bound= {0.98*x3D2, 1.02*x3D2};
        vector<float> xBound2 (bound, bound+2); //todo: to configurable, assume the deviation in (R,t) estimation has 2% portion of accuracy deviation.
         bound= {0.98*y3D2, 1.02*y3D2};
        vector<float> yBound2  (bound, bound+2); //todo: to configurable, assume the deviation in (R,t) estimation has 2% portion of accuracy deviation.
        for(auto & x: xBound2)
           for(auto & y: yBound2)
               {
                    boundPoints.push_back((cv::Mat_<float>(3,1) << x, y, upperZ));
                }
    }
    else
        return false;
    
    //Project to  Neighbor KeyFrames
    
    float upperU = 0.0;
    float upperV = 0.0;
    float lowerU = 0.0;
    float lowerV = 0.0;
    float tempU = 0.0;
    float tempV = 0.0;
    bool valid = false;
    
    //currently only roughly search the area to ensure all deviation are covered in the search area.
    for(auto & bp: boundPoints)
    {
         valid = pKF2->ProjectStereo(bp, tempU, tempV);
         if(!valid) 
                return false;
        if ( tempU > upperU)
            upperU = tempU;
        
        if (tempU < lowerU)
            lowerU = tempU;
            
        if(tempV > upperV)
            upperV = tempV;
            
        if(tempV< lowerV)
            lowerV = tempV;
    }


lowerBoundXInKF2 = lowerU;
lowerBoundYInKF2 = lowerV;
upperBoundXInKF2 = upperU;
upperBoundYInKF2 = upperV;
   
    return true;
    
}