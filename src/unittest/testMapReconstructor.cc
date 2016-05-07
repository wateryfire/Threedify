
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

bool getSearchAreaForWorld3DPointInKF ( KeyFrameStub* const  pKF1, KeyFrameStub* const pKF2, const RcKeyPoint& twoDPoint,float& u0, float& v0, float& u1, float& v1, float& offsetU, float& offsetV );
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
   float u0,v0,u1,v1,offsetU, offsetV;
    bool validResult;
    validResult = getSearchAreaForWorld3DPointInKF (&KF1, &KF2, twoDPoint, u0, v0, u1, v1, offsetU, offsetV);   
    
    

cout << "u0" << " "<<u0 << " ,"<<v0<< "\n" << "u1" << " "<<u1 << " ,"<<v1<< "\n" << "offset" << " "<<offsetU<< " ,"<<offsetV<< "\n"<<endl;

    return 0;
}



 bool getSearchAreaForWorld3DPointInKF ( KeyFrameStub* const  pKF1, KeyFrameStub* const pKF2, const RcKeyPoint& twoDPoint,float& u0, float& v0, float& u1, float& v1, float& offsetU, float& offsetV )
{
    //Uproject lower and upper point from KF1 to world
     //Uproject lower and upper point from KF1 to world
    const float z = twoDPoint.mDepth;
    const float xDelta = 0.01*z, yDelta = 0.01*z;
    cv::Mat P3DcEst = cv::Mat();
    cv::Mat lower3Dw = cv::Mat();
     cv::Mat upper3Dw = cv::Mat();
      cv::Mat KF1Twc = cv::Mat();
     vector<cv::Mat>  boundPoints;
     boundPoints.reserve(4);  //todo: to configurable, considering the deviation in (R,t), in depth so it has 3d distribution cubic.
    float uTho, vTho;  //Point cord after project to KF2
  
    if(z>1 && z<8)  //todo: to configurable, depth <= 1m is not good for RGBD sensor, depth >=8 m cause the depth distribution not sensitive.
    {

        float ZcBound[] = {0.95*z, 1.05*z};  //todo: to configurable, assume the depth estimation has 5% portion of accuracy deviation
   
        const float u = twoDPoint.pt.x;
        const float v = twoDPoint.pt.y;
  
       P3DcEst  = pKF1->UnprojectToCameraCoord(u,v,z);
       KF1Twc = pKF1->GetPoseInverse();
       
        float XcEst = P3DcEst.at<float>(0);
        float YcEst = P3DcEst.at<float>(1);
 

        
        cv::Mat P3Dc0 = (cv::Mat_<float>(3,1) <<XcEst, YcEst, ZcBound[0]);
        cv::Mat P3Dw0=KF1Twc.rowRange(0,3).colRange(0,3)*P3Dc0+KF1Twc.rowRange(0,3).col(3);
        bool valid = pKF2->ProjectStereo(P3Dw0, u0, v0);
        if (!valid)
            return false;

        cv::Mat P3Dc1 = (cv::Mat_<float>(3,1) <<XcEst, YcEst, ZcBound[1]);
        cv::Mat P3Dw1=KF1Twc.rowRange(0,3).colRange(0,3)*P3Dc1+KF1Twc.rowRange(0,3).col(3);
        valid = pKF2->ProjectStereo(P3Dw1 , u1, v1);
        if (!valid)
            return false;
        
        uTho = (u0+u1)/2;
        vTho =(v0+v1)/2;
     
//        cout <<"Xc estimation: "<< XcEst << " Yc estimation:"<<YcEst << "  Zc estimation"<< z<<endl;
        float XcBound[]= {XcEst-xDelta, XcEst+xDelta};
        float YcBound[]= { YcEst-yDelta,  YcEst+yDelta};
       for ( int xindex = 0; xindex < 2; xindex ++)
       {
           cv::Mat P3Dc = (cv::Mat_<float>(3,1) <<XcBound[xindex], YcEst, z);
                    cv::Mat P3Dw=KF1Twc.rowRange(0,3).colRange(0,3)*P3Dc+KF1Twc.rowRange(0,3).col(3);
             
                 
                    boundPoints.push_back( P3Dw);
       }
       for ( int yindex = 0; yindex < 2; yindex ++)
       {
           cv::Mat P3Dc = (cv::Mat_<float>(3,1) <<XcEst, YcBound[yindex], z);
           cv::Mat P3Dw=KF1Twc.rowRange(0,3).colRange(0,3)*P3Dc+KF1Twc.rowRange(0,3).col(3);
            boundPoints.push_back( P3Dw);
       }

         

    }
    else
        return false;
    
    //Project to  Neighbor KeyFrames
    
    float maxOffsetU = 0.0;
    float maxOffsetV =0.0;
    float tempOffsetU = 0.0;
    float tempOffsetV = 0.0;
    float tempU = 0.0;
    float tempV = 0.0;

    
    //currently only roughly search the area to ensure all deviation are covered in the search area.
//    cout <<"bound points"<<endl;
    for(auto & bp: boundPoints)
    {
         bool valid = pKF2->ProjectStereo(bp, tempU, tempV);
         if(!valid) 
                return false;
         tempOffsetU = fabs(tempU-uTho);
         tempOffsetV = fabs(tempV-vTho);
//         cout << tempU<< "  " <<       tempV << endl;

        if (  tempOffsetU > maxOffsetU )
            maxOffsetU =tempOffsetU;
        
      if (  tempOffsetV> maxOffsetV )
            maxOffsetV =tempOffsetV;
            
    }


    offsetU = maxOffsetU;
    offsetV = maxOffsetV;
   
    return true;
    
}


