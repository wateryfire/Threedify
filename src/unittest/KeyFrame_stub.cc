/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "KeyFrame_stub.h"

#include <mutex>

namespace ORB_SLAM2
{



KeyFrameStub::KeyFrameStub():
   mnMinX(0), mnMinY(0), mnMaxX(640),
    mnMaxY(480), cx(316.4106496157017), cy(239.6702484140453),fx(540.3364232281056), fy(539.127143995656), invfx(1.0/fx),invfy(1.0/fy)
{

    Tcw = cv::Mat::eye(4,4,CV_32F);
   Ow = cv::Mat::eye(3,3,CV_32F);
   
}



void KeyFrameStub::SetPose(const cv::Mat &Tcw_)
{

    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));

}



bool KeyFrameStub::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}



bool KeyFrameStub::ProjectStereo(cv::Mat& x3Dw, float& u, float& v)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);

    const cv::Mat Pc = Rcw*x3Dw+tcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;
    
    return true;
}

cv::Mat KeyFrameStub::UnprojectStereo(float u,float v, float z)
{
    if(z>0)
    {
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);


        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

cv::Mat KeyFrameStub::UnprojectToCameraCoord(float u,float v, float z)
{
    if(z>0)
    {
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        return x3Dc;
    }
    else
        return cv::Mat();
}

cv::Mat KeyFrameStub::GetPoseInverse()
{
    return Twc.clone();
}


} //namespace ORB_SLAM
