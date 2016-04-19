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

        // hyposis inverse depth
        float tho;
        // hyposis inverse depth standard
        float sigma;

        // status
        bool hasHypo;
        bool fused;

        // color if needed
        long rgb;

        /**
       * Constructor
       */
        RcKeyPoint(float x, float y, float intensity, float gradient, float orientation,int octave,float depth):
            pt(x,y), intensity(intensity), gradient(gradient),orientation(orientation),octave(octave),mDepth(depth),hasHypo(false),fused(false){}

        // with depth sensor
        void setMDepth(float depth){
            mDepth = depth;
        }

        void updateHypo(float thoStar, float sigmaStar){
            tho = thoStar;
            sigma = sigmaStar;
            hasHypo = true;
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


    map<long, vector<RcKeyPoint>> keyframeKeyPointsMap;

    bool CheckNewKeyFrames(KeyFrame* currentKeyFrame);
    void CreateNewMapPoints(KeyFrame* currentKeyFrame);

    void highGradientAreaKeyPoints(cv::Mat &gradient, cv::Mat &orientation, KeyFrame *pKF, const float gradientThreshold);
    void getApproximateOctave(KeyFrame *pKF,std::map<int,pair<float, float>> &octaveDepthMap);

    cv::Mat UnprojectStereo(RcKeyPoint &p,KeyFrame *pKF);
    cv::Mat ComputeF12(KeyFrame *pKF1, KeyFrame *pKF2);
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
    void epipolarConstraientSearch(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12, vector<pair<size_t,size_t> > &vMatchedIndices);
    float calInverseDepthEstimation(RcKeyPoint &kp1,const float u0Star,KeyFrame *pKF1, KeyFrame *pKF2);
    bool CheckDistEpipolarLine(RcKeyPoint& kp1,RcKeyPoint& kp2,cv::Mat &F12,KeyFrame* pKF2);
    float calcMedianRotation(KeyFrame* pKF1, KeyFrame* pKF2);
    float calcInPlaneRotation(KeyFrame* pKF1, KeyFrame* pKF2);

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
