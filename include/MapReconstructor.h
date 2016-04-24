/*
 * MapReconstructor.h
 *
 *  Created on: Apr 11, 2016
 *      Author: qiancan
 */

#ifndef INCLUDE_MAPRECONSTRUCTOR_H_
#define INCLUDE_MAPRECONSTRUCTOR_H_

#include "System.h"

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

    bool realTimeReconstructionEnd = false;

    // re-construction key point
    struct RcKeyPoint
    {
        cv::Point2f pt;
        float intensity;
        float gradient;
        float orientation;
        int octave;

        // depth from mesurements
        float mDepth;

        // inverse depth hypotheses set
        vector<pair<float, float>> hypotheses;
        map<RcKeyPoint*, int> hypothesesRelation;

        // final hypothese
        float tho = NAN;
        float sigma = NAN;

        // status
        bool hasHypo;
        bool fused = false;

        // color if needed
        long rgb;

        /**
       * Constructor
       */
        RcKeyPoint(){}

        RcKeyPoint(float x, float y, float intensity, float gradient, float orientation,int octave,float depth):
            pt(x,y), intensity(intensity), gradient(gradient),orientation(orientation),octave(octave),mDepth(depth),hasHypo(false),fused(false){}

        // with depth sensor
        void setMDepth(float depth){
            mDepth = depth;
        }

        void addHypo(float thoStar, float sigmaStar, RcKeyPoint* match){
            hypothesesRelation[match] = hypotheses.size();
            hypotheses.push_back(make_pair(thoStar, sigmaStar));
            hasHypo = true;
        }
    };
    struct Point2fLess
    {
        bool operator()(cv::Point2f const&lhs, cv::Point2f const& rhs) const
        {
            return lhs.x == rhs.x ? lhs.y < rhs.y : lhs.x < rhs.x;
        }
    };

public:

    MapReconstructor(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc, Tracking* pTracker, const string & strSettingsFile);

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
    int kN = 7;
    // intensity standard deviation
    float sigmaI = 20.0;
    // high gradient threshold
    float lambdaG = 8.0;
    // epipolar line angle threshold
    float lambdaL = 80.0;
    // orientation angle threshold
    float lambdaThe = 45.0;
    // compactibility size threshold
    int lambdaN = 3;
    // noise relation factor
    float theta = 0.23;

    // key points for epipolar search
    map<long, map<cv::Point2f,RcKeyPoint,Point2fLess> > keyframeKeyPointsMap;

    bool CheckNewKeyFrames(KeyFrame* currentKeyFrame);
    void CreateNewMapPoints(KeyFrame* currentKeyFrame);

    void highGradientAreaKeyPoints(cv::Mat &gradient, cv::Mat &orientation, KeyFrame *pKF, const float gradientThreshold);
    void getApproximateOctave(KeyFrame *pKF,std::map<int,pair<float, float>> &octaveDepthMap);

    cv::Mat UnprojectStereo(RcKeyPoint &p,KeyFrame *pKF);
    cv::Mat ComputeF12(KeyFrame *pKF1, KeyFrame *pKF2);
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
    void epipolarConstraientSearch(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12, vector<pair<size_t,size_t> > &vMatchedIndices);
    bool calCordBounds(cv::Point2f &startCordRef, cv::Point2f &endCordRef, float width, float height, float a, float b,float c);
    float checkEpipolarLineConstraient(RcKeyPoint &kp1, RcKeyPoint &kp2, float a, float b, float c, float medianRotation, KeyFrame *pKF2);
    float calInverseDepthEstimation(RcKeyPoint &kp1,const float u0Star,KeyFrame *pKF1, KeyFrame *pKF2);
    bool CheckDistEpipolarLine(RcKeyPoint& kp1,RcKeyPoint& kp2,cv::Mat &F12,KeyFrame* pKF2);
    float calcMedianRotation(KeyFrame* pKF1, KeyFrame* pKF2);
    float calcInPlaneRotation(KeyFrame* pKF1, KeyFrame* pKF2);
    void calcSenceInverseDepthBounds(KeyFrame* pKF, float &tho0, float &sigma0);
    bool cordInImageBounds(float x, float y, int width, int height);

    void fuseHypo(KeyFrame* pKF);

private:

	//Extract and store the edge profile info for a key frame
	void ExtractEdgeProfile(KeyFrame *pKeyFrame);

    Map* mpMap;
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;
    Tracking* mpTracker;

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
