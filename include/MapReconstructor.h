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
    bool realTimeReconstructionEnd = false;

    // cached camera params
    int width;
    int height;

    // Dist Coef
    cv::Mat mDistCoef;

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

        // neighbour pixels, indexed 0-8 in clock order, stors intensity and gradient
        vector<vector<float>> neighbours;

        // inverse depth hypotheses set
        vector<pair<float, float>> hypotheses;
        map<RcKeyPoint*, int> hypothesesRelation;

        // final hypothese
        float tho = NAN;
        float sigma = NAN;

        // status
        bool hasHypo;
        bool fused = false;

        // check status
        int intraCheckCount = 0;
        int interCheckCount = 0;

        // project neighbour error factors, list of djn, rjiz*xp, tjiz, sigmajnSquar
//        vector<vector<float>> depthErrorEstimateFactor;

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

        // store neighbour info for epipolar search
        void fetchNeighbours(cv::Mat &image, cv::Mat &gradient)
        {
            /*eachNeighbourCords([&](cv::Point2f ptn){
                vector<float> msg;
                msg.push_back(image.at<float>(ptn));
                msg.push_back(gradient.at<float>(ptn));
                neighbours.push_back(msg);
            });*/
            // pixel on edge check ?
            const float degtorad = M_PI/180;  // convert degrees
            for(int i=0;i<360;i+=45)
            {
                int dx = round(cos(i*degtorad));
                int dy = round(sin(i*degtorad));
                cv::Point cur = cv::Point(pt.x+dx, pt.y+dy);
                vector<float> msg;
                msg.push_back(image.at<float>(cur));
                msg.push_back(gradient.at<float>(cur));

                neighbours.push_back(msg);
            }
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
                int dx = round(cos(i*degtorad));
                int dy = round(sin(i*degtorad));
                cv::Point2f cur = cv::Point2f(pt.x+dx, pt.y+dy);
                func(cur);
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

    // additional params (DEBUG)
    float depthThresholdMax;
    float depthThresholdMin;
    float epipolarSearchOffset;

    // key points for epipolar search
    map<long, map<cv::Point2f,RcKeyPoint,Point2fLess> > keyframeKeyPointsMap;

    bool CheckNewKeyFrames(KeyFrame* currentKeyFrame);
    void CreateNewMapPoints(KeyFrame* currentKeyFrame);

    void highGradientAreaKeyPoints(cv::Mat &gradient, cv::Mat &orientation, KeyFrame *pKF, const float gradientThreshold);
    //void getApproximateOctave(KeyFrame *pKF,std::map<int,pair<float, float>> &octaveDepthMap);

    cv::Mat UnprojectStereo(RcKeyPoint &p,KeyFrame *pKF);
    cv::Mat ComputeF12(KeyFrame *pKF1, KeyFrame *pKF2);
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
    void epipolarConstraientSearch(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12, vector<pair<size_t,size_t> > &vMatchedIndices);
//    float MatchAlongEpipolarLine(cv::Point2f &matchedCord, RcKeyPoint &kp1, map<cv::Point2f,RcKeyPoint,Point2fLess> &keyPoints2, float &medianRotation, float &u0 ,float &u1, float &v0, float &v1, const float &a, const float &b, const float &c);
    float MatchAlongEpipolarLine(cv::Point2f &matchedCord, RcKeyPoint &kp1, cv::SparseMat_<RcKeyPoint*> &keyPoints2, float &medianRotation, float &u0 ,float &u1, float &v0, float &v1, const float &a, const float &b, const float &c);
    bool calCordBounds(cv::Point2f &startCordRef, cv::Point2f &endCordRef, float width, float height, float a, float b,float c);
    float checkEpipolarLineConstraient(RcKeyPoint &kp1, RcKeyPoint &kp2, float a, float b, float c, float medianRotation);
    float calInverseDepthEstimation(RcKeyPoint &kp1,const float u0Star,KeyFrame *pKF1, KeyFrame *pKF2);
    //bool CheckDistEpipolarLine(RcKeyPoint& kp1,RcKeyPoint& kp2,cv::Mat &F12,KeyFrame* pKF2);
    float calcMedianRotation(KeyFrame* pKF1, KeyFrame* pKF2);
    //float calcInPlaneRotation(KeyFrame* pKF1, KeyFrame* pKF2);
    void calcSenceInverseDepthBounds(KeyFrame* pKF, float &tho0, float &sigma0);
    bool cordInImageBounds(float x, float y, int width, int height);

    void fuseHypo(KeyFrame* pKF);
    int KaTestFuse(std::vector<std::pair<float, float>> &hypos, float &tho, float &sigma, set<int> &nearest);
    void intraKeyFrameChecking(KeyFrame* pKF);
    void addKeyPointToMap(RcKeyPoint &kp1, KeyFrame* pKF);
    void interKeyFrameChecking(KeyFrame* pKF);
    void Distort(cv::Point2f &point, KeyFrame* pKF);
    
    bool  getSearchAreaForWorld3DPointInKF( KeyFrame* const  pKF1, KeyFrame* const pKF2, const RcKeyPoint& twoDPoint,float& u0, float& v0, float& u1, float& v1, float& offsetU, float& offsetV );
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
