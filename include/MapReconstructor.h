/*
 * MapReconstructor.h
 *
 *  Created on: Apr 11, 2016
 *      Author: qiancan
 */

#ifndef INCLUDE_MAPRECONSTRUCTOR_H_
#define INCLUDE_MAPRECONSTRUCTOR_H_

#include<string>
#include<thread>
#include<mutex>
#include<unistd.h>
#include<opencv2/core/core.hpp>

#include "Map.h"

namespace ORB_SLAM2
{

class MapReconstructor
{
public:

	//All defined status of job
    enum JobStatus{
        INITIALIZED = 0,
		STARTED,
        STOPPED
    };

    //TODO add mb prefix
    bool mRealTimeReconstructionEnd = false;

    // cached camera params
    int mWidth;
    int mHeight;

    // Dist Coef
    cv::Mat mDistCoef;

    // re-construction key point
    struct RcHighGradientPoint
    {
        cv::Point2f pt;
        float intensity;
        float gradient;
        float orientation;
        int octave;

        // depth from mesurements
        float mDepth;

        // neighbour pixels, indexed 0-8 in clock order, stors intensity and gradient
        vector<vector<float>> neighbours;

        // inverse depth hypotheses set
        vector<pair<float, float>> hypotheses;
        // indexed relation of hypotheses
        map<RcHighGradientPoint*, int> hypothesesRelation;

        // final hypothese
        float tho = NAN;
        float sigma = NAN;

        // status
        bool hasHypo;
        bool fused = false;

        // check status
        int intraCheckCount = 0;
        int interCheckCount = 0;

        /**
       * Constructor
       */
        RcHighGradientPoint(){}

        RcHighGradientPoint(float x, float y, float intensity, float gradient, float orientation,int octave,float depth):
            pt(x,y), intensity(intensity), gradient(gradient),orientation(orientation),octave(octave),mDepth(depth),hasHypo(false),fused(false){}

        // with depth sensor
        void setMDepth(float depth)
        {
            mDepth = depth;
        }

        void addHypo(float thoStar, float sigmaStar, RcHighGradientPoint* match)
        {
//            hypothesesRelation[match] = hypotheses.size();
            hypotheses.push_back(make_pair(thoStar, sigmaStar));
            hasHypo = true;
        }

        // get two neighbours on line ax + by + c = 0
        void getNeighbourAcrossLine(const float a, const float b, vector<float> &lower, vector<float> &upper)
        {
            //
            float angle = cv::fastAtan2(-a,b);
            int index = 0;
            for(int i=0;i<360;i+=45)
            {
                if(abs(angle-i) <= 22.5)
                {
                    break;
                }
                index++;
            }
            index %=8;
            if(index>=3 && index<=6)
            {
                index = (index+4)%8;
            }

            upper = neighbours[index];
            lower = neighbours[(index+4)%8];
        }

        template <typename Func>
        void eachNeighbourCords(Func const& func)
        {
            const float degtorad = M_PI/180;  // convert degrees
            for(int i=0;i<360;i+=45)
            {
                float nx = pt.x + round(cos((float)i*degtorad));
                float ny = pt.y + round(sin((float)i*degtorad));
                func(nx, ny);
            }
        }
    };
    struct Point2fLess
    {
        bool operator()(cv::Point2f const&lhs, cv::Point2f const& rhs) const
        {
            return (lhs.x == rhs.x) ? (lhs.y < rhs.y) : (lhs.x < rhs.x);
        }
    };

public:

    MapReconstructor(Map* pMap, const string & strSettingsFile);

	//Add a new key frame into the queue to be processed
	void InsertKeyFrame(KeyFrame *pKeyFrame);

	//Main function to process key frames in the queue
	void RunToProcessKeyFrameQueue();

	//Main function to process the map reconstruction
	void RunToReconstructMap();

	void StartKeyFrameQueueProcess();
	void StopKeyFrameQueueProcess();

	void StartRealTimeMapReconstruction();
	void StopRealTimeMapReconstruction();
    bool isRealTimeReconstructionEnd();

	void StartFullMapReconstruction();
    void StopFullMapReconstruction();

    // reconstruction params
    // keyframes set size N
    int mKN = 7;
    // intensity standard deviation
    float mSigmaI = 20.0;
    // high gradient threshold
    float mLambdaG = 8.0;
    // epipolar line angle threshold
    float mLambdaL = 80.0;
    // orientation angle threshold
    float mLambdaThe = 45.0;
    // compactibility size threshold
    int mLambdaN = 3;
    // noise relation factor
    float mTheta = 0.23;

    // additional params (DEBUG)
    float mDepthThresholdMax;
    float mDepthThresholdMin;
    float mEpipolarSearchOffset;

    // key points for epipolar search
    map<long, map<cv::Point2f,RcHighGradientPoint,Point2fLess> > mKeyframeKeyPointsMap;

    bool CheckNewKeyFrames(KeyFrame* currentKeyFrame);
    void CreateNewMapPoints(KeyFrame* currentKeyFrame);

    void HighGradientAreaPoints(cv::Mat &gradient, cv::Mat &orientation, KeyFrame *pKF, const float gradientThreshold);
    std::set<cv::Point,Point2fLess> DepthCurvatureFilter(cv::Mat &depths);
    //void getApproximateOctave(KeyFrame *pKF,std::map<int,pair<float, float>> &octaveDepthMap);

    cv::Mat UnprojectStereo(RcHighGradientPoint &p,KeyFrame *pKF);
    cv::Mat ComputeF12(KeyFrame *pKF1, KeyFrame *pKF2);
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
    void EpipolarConstraientSearch(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12, vector<pair<size_t,size_t> > &vMatchedIndices);
//    float MatchAlongEpipolarLine(cv::Point2f &matchedCord, RcKeyPoint &kp1, map<cv::Point2f,RcKeyPoint,Point2fLess> &keyPoints2, float &medianRotation, float &u0 ,float &u1, float &v0, float &v1, const float &a, const float &b, const float &c);
    float MatchAlongEpipolarLine(cv::Point2f &matchedCord, RcHighGradientPoint &kp1, cv::SparseMat_<RcHighGradientPoint*> &keyPoints2, float &medianRotation, float &u0 ,float &u1, float &v0, float &v1, const float &a, const float &b, const float &c);
    bool CalCordBounds(cv::Point2f &startCordRef, cv::Point2f &endCordRef, float mWidth, float mHeight, float a, float b,float c);
    float CheckEpipolarLineConstraient(RcHighGradientPoint &kp1, RcHighGradientPoint &kp2, float a, float b, float c, float medianRotation);
    float CalInverseDepthEstimation(RcHighGradientPoint &kp1,const float u0Star,KeyFrame *pKF1, KeyFrame *pKF2);
    //bool CheckDistEpipolarLine(RcKeyPoint& kp1,RcKeyPoint& kp2,cv::Mat &F12,KeyFrame* pKF2);
    float CalcMedianRotation(KeyFrame* pKF1, KeyFrame* pKF2);
    //float calcInPlaneRotation(KeyFrame* pKF1, KeyFrame* pKF2);
    void CalcSenceInverseDepthBounds(KeyFrame* pKF, float &tho0, float &sigma0);
    bool CordInImageBounds(float x, float y, int mWidth, int mHeight);

    void FuseHypo(KeyFrame* pKF);
    int KaTestFuse(std::vector<std::pair<float, float>> &hypos, float &tho, float &sigma, set<int> &nearest);
    void IntraKeyFrameChecking(KeyFrame* pKF);
    void AddPointToMap(RcHighGradientPoint &kp1, KeyFrame* pKF);
    void InterKeyFrameChecking(KeyFrame* pKF);
    void Distort(cv::Point2f &point, KeyFrame* pKF);
    
    bool  GetSearchAreaForWorld3DPointInKF( KeyFrame* const  pKF1, KeyFrame* const pKF2, const RcHighGradientPoint& twoDPoint,float& u0, float& v0, float& u1, float& v1, float& offsetU, float& offsetV );
private:

	//Extract and store the edge profile info for a key frame
	void ExtractEdgeProfile(KeyFrame *pKeyFrame);

    Map* mpMap;

    std::list<KeyFrame*> mlpKeyFrameQueue;
    std::mutex mMutexForKeyFrameQueue;

    std::list<KeyFrame*> mlpKFQueueForReonstruction;
    std::mutex mMutexForKFQueueForReonstruction;

	//To indicate the current status of jobs.
	JobStatus mStatus_KeyFrameQueueProcess;
	JobStatus mStatus_RealTimeMapReconstruction;
	JobStatus mStatus_FullMapReconstruction;

};

} //namespace ORB_SLAM2


#endif /* INCLUDE_MAPRECONSTRUCTOR_H_ */
