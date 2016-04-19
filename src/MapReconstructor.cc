/*
 * MapReconstructor.cc
 *
 *  Created on: Apr 11, 2016
 *      Author: qiancan
 */

#include "MapReconstructor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

MapReconstructor::MapReconstructor(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc, Tracking* pTracker,  const string &strSettingPath):
		mpMap(pMap), mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpTracker(pTracker)
{
	mStatus_KeyFrameQueueProcess=INITIALIZED;
	mStatus_RealTimeMapReconstruction=INITIALIZED;
	mStatus_FullMapReconstruction=INITIALIZED;

    // Get re-construction params from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    kN = fSettings["ReConstruction.KN"];
    sigmaI = fSettings["ReConstruction.sigmaI"];
    lambdaG = fSettings["ReConstruction.lambdaG"];
    lambdaL = fSettings["ReConstruction.lambdaL"];
    lambdaThe = fSettings["ReConstruction.lambdaThe"];
    lambdaN = fSettings["ReConstruction.lambdaN"];
    theta = fSettings["ReConstruction.theta"];
}

void MapReconstructor::InsertKeyFrame(KeyFrame *pKeyFrame)
{
    unique_lock<mutex> lock(mMutexForKeyFrameQueue);
    if(pKeyFrame!=NULL && pKeyFrame->mnId!=0)
    {
    	mlpKeyFrameQueue.push_back(pKeyFrame);
    }
}

void MapReconstructor::RunToProcessKeyFrameQueue()
{
	int nCounter=0;
	KeyFrame* currentKeyFrame=NULL;

	cout << "MapReconstructor: Start thread execution for processing key frames queue." << endl;

	while(mStatus_KeyFrameQueueProcess!=STOPPED)
	{
        usleep(300);

		if(mStatus_KeyFrameQueueProcess!=STARTED)
		{
			continue;
		}

		//Get Mutex-lock to access the queue of key frames.
	    {
	        unique_lock<mutex> lock(mMutexForKeyFrameQueue);
	        if(mlpKeyFrameQueue.empty()==true)
	        {
	        	continue;
	        }

	        currentKeyFrame = mlpKeyFrameQueue.front();
	        mlpKeyFrameQueue.pop_front();
	    }

	    ExtractEdgeProfile(currentKeyFrame);

	    // Release or never release the images?

	    //currentKeyFrame->mRefImgGray.release();
	    //currentKeyFrame->mRefImgDepth.release();

		cout << "MapReconstructor: Release  gray image: " << currentKeyFrame->mRefImgGray.total() << " pixel(s) remain." << endl;
		cout << "MapReconstructor: Release depth image: " << currentKeyFrame->mRefImgGray.total() << " pixel(s) remain." << endl;

		{
			// Add to the second queue for the real-time map reconstruction
		    unique_lock<mutex> lock(mMutexForKFQueueForReonstruction);
		    if(currentKeyFrame!=NULL && currentKeyFrame->mnId!=0)
		    {
		    	mlpKFQueueForReonstruction.push_back(currentKeyFrame);
		    }
		}

	    nCounter++;
	}

	cout << "MapReconstructor: End thread execution for processing key frames queue. Total " << nCounter << " key frames were processed." << endl;
}

void MapReconstructor::ExtractEdgeProfile(KeyFrame *pKeyFrame)
{
	cout << "MapReconstructor: Extracting edge profile info from the key frame (FrameId: " << pKeyFrame->mnId << ")." << endl;

	if(pKeyFrame==NULL || pKeyFrame->mRefImgGray.empty() || pKeyFrame->mRefImgDepth.empty())
    {
    	return;
    }

	cout << "MapReconstructor: Processing  gray image: " << pKeyFrame->mRefImgGray.total() << " pixel(s)." << endl;
	cout << "MapReconstructor: Processing depth image: " << pKeyFrame->mRefImgDepth.total() << " pixel(s)." << endl;

    // gradient / orientation
    Mat gradientX, gradientY;
    Scharr( pKeyFrame->mRefImgGray, gradientX , CV_32F, 1, 0);
    Scharr( pKeyFrame->mRefImgGray, gradientY, CV_32F, 0, 1);

    // Cartezian -> Polar
    Mat modulo;
    Mat orientation;
    cartToPolar(gradientX,gradientY,modulo,orientation,true);

    // ? loss of precies
    normalize(modulo, modulo, 0x00, 0xFF, NORM_MINMAX, CV_8U);

    // ? stor opt
    pKeyFrame->mRefImgGradient = modulo;

    highGradientAreaKeyPoints(modulo,orientation, pKeyFrame, lambdaG);
}

void MapReconstructor::highGradientAreaKeyPoints(Mat &gradient, Mat &orientation, KeyFrame *pKF, const float gradientThreshold){
    Mat image = pKF->mRefImgGray;
    Mat depths = pKF->mRefImgDepth;

     vector<RcKeyPoint> keyPoints;

    // map feature key points to octave, yield depth intervals for octave
    map<int, pair<float, float> > octaveDepthMap;
    getApproximateOctave(pKF, octaveDepthMap);
    map<int, pair<float, float> >::iterator  iter;

    ///////////////////////
//    vector<float> ocv  = pKF->mvLevelSigma2;
//    vector<float>::iterator oci;
//    for(oci = ocv.begin(); oci != ocv.end(); oci++)
//    {
//        cout<< "octave sigma: " <<*oci<<endl;
//    }

//    for(iter = octaveDepthMap.begin(); iter != octaveDepthMap.end(); iter++)
//    {
//        cout<< "octave map: " <<iter->first <<" "<<iter->second.first<<" "<<iter->second.second<<endl;
//    }

    /////////////////////

    int eraseCount =0;
    for(int row = 0; row < gradient.rows; ++row) {
        uchar* p = gradient.ptr<uchar>(row);
        for(int col = 0; col < gradient.cols; ++col) {
            int gradientModulo = p[col];

            // 1. filter with intensity threshold
            if(gradientModulo <= gradientThreshold)
            {
                eraseCount++;
                continue;
            }

            // 2. estimate octave
            // judge which interval of one point on edge belongs to, give an approximate estimation of octave
            int octave;
            float depth = depths.at<float>(row, col);
            for(iter = octaveDepthMap.end(); iter != octaveDepthMap.begin(); iter--)
            {
                octave = iter->first;
                if(depth > iter->second.second && depth<=iter->second.second){
                    break;
                }
            }
            //
            float angle = orientation.at<float>(row, col);
            float intensity = image.at<float>(row, col);
            float gradient = gradientModulo;
            keyPoints.push_back(RcKeyPoint(row,col,intensity,gradient,angle,octave,depth));
        }
    }
    keyframeKeyPointsMap[pKF->mnId] = keyPoints;
}

void MapReconstructor::getApproximateOctave(KeyFrame *pKF,map<int,pair<float, float>> &octaveDepthMap){
    vector<KeyPoint> keyPoints = pKF->mvKeysUn;
    vector<float> depths = pKF->mvDepth;

    int size = (int)keyPoints.size();
    for(int i=0;i<size;i++){
        KeyPoint p = keyPoints[i];
        float depth = depths[i];

        if(depth<0){continue;}

        int octave = p.octave;

        std::pair<float, float> interval;

        float sup,sub;
        if(octaveDepthMap.count(octave)<1){
            interval = make_pair(NAN, NAN);
            sub = NAN;
            sup = NAN;
        }
        else{
            interval = octaveDepthMap[octave];
            sub = interval.first;
            sup = interval.second;
        }

        if(isnan(sup) && isnan(sub)){
            sup = sub = depth;
        }
        else{
            if(depth<sup){
                if(depth<sub){
                    sub = depth;
                }
            }else{
                sup = depth;
            }
        }
        interval.first = sub;
        interval.second = sup;

        octaveDepthMap[octave] = interval;
    }

    /*map<int, std::pair<float, float> >::iterator  iter;
    // grow invervals to cover the whole possitive axis
    float upper = FLT_MAX,lower = NAN;
    for(iter = octaveDepthMap.begin(); iter != octaveDepthMap.end(); iter++)
    {
        if(!isnan(lower)){
            iter->second.first = lower;
        }
        lower = iter->second.second;
    }
    iter--;
    iter->second.second = upper; // to possitive infinate*/
}

void MapReconstructor::RunToReconstructMap()
{
    KeyFrame* currentKeyFrame=NULL;

	while(mStatus_RealTimeMapReconstruction!=STARTED)
	{
        usleep(1000);
	}

	cout << "MapReconstructor: Start thread execution for map reconstruction during SLAM tracking." << endl;

    int retryCount = 0;
    while(mStatus_RealTimeMapReconstruction!=STOPPED)
    {
        if(mlpKFQueueForReonstruction.empty())
        {
            usleep(1000);

            continue;
        }

        currentKeyFrame = mlpKFQueueForReonstruction.front();
        cout << "MapReconstructor: Reconstructing map from the key frame (FrameId: " << currentKeyFrame->mnId << ")." << endl;

        bool frameValid = CheckNewKeyFrames(currentKeyFrame);
        if (frameValid || retryCount>10)
        {
            //Get Mutex-lock to access the queue of key frames.
            {
                unique_lock<mutex> lock(mMutexForKFQueueForReonstruction);
                mlpKFQueueForReonstruction.pop_front();
            }
            retryCount=0;
        }
        else
        {
            retryCount ++;
            usleep(1000);
        }

        if(frameValid) {
            CreateNewMapPoints(currentKeyFrame);
        }
    }

	cout << "MapReconstructor: End thread execution for map reconstruction during SLAM tracking." << endl;

	cout << "MapReconstructor: Start thread execution for full map reconstruction." << endl;

	// TODO: Remove the sleep process, once the real code is implemented.
//	usleep(10000);

	cout << "MapReconstructor: End thread execution for full map reconstruction." << endl;
}

bool MapReconstructor::CheckNewKeyFrames(KeyFrame* currentKeyFrame)
{
    const vector<KeyFrame*> vpNeighKFs = currentKeyFrame->GetBestCovisibilityKeyFrames(kN);
    return vpNeighKFs.size() >= kN;
}

void MapReconstructor::CreateNewMapPoints(KeyFrame* mpCurrentKeyFrame)
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = kN;
//    if(mbMonocular)
//        nn=2 * kN;

    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    int nnew=0;

    vector<RcKeyPoint> &keyPoints1 = keyframeKeyPointsMap[mpCurrentKeyFrame->mnId];

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
//        if(i>0 && CheckNewKeyFrames())
//            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        long mnid2 = pKF2->mnId;
        // while ? TODO
        if(!keyframeKeyPointsMap.count(mnid2)){
            cout << "keyframe data not extracted yet: " << mnid2 << endl;
            continue;
        }

//        vector<RcKeyPoint> &keyPoints2 = keyframeKeyPointsMap[pKF2->mnId];
        epipolarConstraientSearch(mpCurrentKeyFrame, pKF2, F12,vMatchedIndices);

        cv::Mat Rcw2 = pKF2->GetRotation();
//        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            pair<size_t,size_t> pa = vMatchedIndices[ikp];

            cv::Mat x3D = UnprojectStereo(keyPoints1[pa.first], mpCurrentKeyFrame);
            if(x3D.empty()){
                continue;
            }

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

//            pMP->AddObservation(mpCurrentKeyFrame,idx1);
//            pMP->AddObservation(pKF2,idx2);

//            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
//            pKF2->AddMapPoint(pMP,idx2);

//            pMP->ComputeDistinctiveDescriptors();

//            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
//            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

cv::Mat MapReconstructor::UnprojectStereo(RcKeyPoint &p,KeyFrame *pKF)
{
    const float z = p.mDepth;
//        const float z = 1.0/p.tho;
    const float u = p.pt.x;
    const float v = p.pt.y;
    return pKF->UnprojectStereo(v,u,z);
}

cv::Mat MapReconstructor::ComputeF12(KeyFrame *pKF1, KeyFrame *pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

cv::Mat MapReconstructor::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void MapReconstructor::epipolarConstraientSearch(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,vector<pair<size_t,size_t> > &vMatchedIndices)
{
    vMatchedIndices.clear();
    vector<RcKeyPoint> &keyPoints1 = keyframeKeyPointsMap[pKF1->mnId];
    vector<RcKeyPoint> &keyPoints2 = keyframeKeyPointsMap[pKF2->mnId];

    int keyPoints1Count = keyPoints1.size(),  keyPoints2Count = keyPoints2.size();

    for (int index1 = 0; index1 < keyPoints1Count; ++index1) {
        RcKeyPoint &kp1 = keyPoints1[index1];

        if(kp1.hasHypo){
            continue;
        }

        // prepare data
        float intensity1 = kp1.intensity;
        float angle1 = kp1.orientation;
        float gradient1 = kp1.gradient;

        // epipolar line
        const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
        const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
        const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

        float minSimilarityError = -1.0;
        int matchedKeyPointIdx = -1;

        for (int index2 = 0; index2 < keyPoints2Count; ++index2) {
            RcKeyPoint &kp2 = keyPoints2[index2];

            if(kp2.hasHypo){
                continue;
            }

            // prepare data
            float intensity2 = kp2.intensity;
            float angle2 = kp2.orientation;
            float gradient2 = kp2.gradient;

            // search on epipolar line
            if(!CheckDistEpipolarLine(kp1,kp2,F12,pKF2)){
                continue;
            }

            // check epipolar line angle with orientation

            float eplAngle = fastAtan2(-b,a);
            float eplAngleDiff  = angle2 - eplAngle ;
            if(eplAngleDiff < -0.0){
                eplAngleDiff += 360.0;
            }else if(eplAngleDiff > 360.0){
                eplAngleDiff -= 360.0;
            }
            if(eplAngleDiff >= lambdaL){
                continue;
            }

            // check in-plane rotation
            float medianRotation = calcInPlaneRotation(pKF1,pKF2);
            float angleDiff = angle2 - angle1 - medianRotation;
            if(angleDiff < -0.0){
                angleDiff += 360.0;
            }else if(angleDiff > 360.0){
                angleDiff -= 360.0;
            }
            if(angleDiff >= lambdaThe){
                continue;
            }

            // cal similarity error
            float intensityError = intensity1 - intensity2;
            float gradientError = gradient1 - gradient2;
            float similarityError = (intensityError * intensityError + 1.0/theta * gradientError * gradientError ) / ( sigmaI * sigmaI );

            if(minSimilarityError < 0 || minSimilarityError > similarityError){
                minSimilarityError = similarityError;
                matchedKeyPointIdx = index2;
            }
        }

        // find the best match point
        if(matchedKeyPointIdx >=0){
            RcKeyPoint match = keyPoints2[matchedKeyPointIdx];

            // subpixel estimation:
            // approximate intensity gradient along epipolar line : g=(I(uj + 1)-I(uj - 1))/2;
            // approximate intensity gradient module derivate along epipolar line : q=(G(uj + 1)-G(uj - 1))/2;
            // pixel estimation:
            // u0s = u0 + (g(u0)*ri(u0) + 1/theta * q(u0) * rg(u0)) / (g(u0) * g(u0) + 1/theta * q(u0) * q(u0) )
            // su0s^2 = 2* si*si / (g(u0) * g(u0) + 1/theta * q(u0) * q(u0))

            cv::Mat imageKf2 = pKF2->mRefImgGray, gradientKf2 = pKF2->mRefImgGradient;
            float u0 = match.pt.x;

            // neighbour coordinates of this point
            float u0fUpper = match.pt.x + 1.0, v0fUpper = - (a*u0fUpper-c)/b;
            int u0Upper = (int)round(u0fUpper), v0Upper = (int)round(v0fUpper);
//            if(u0Upper <-640 || v0Upper<-640 || u0Upper > 640 || v0Upper > 640 ){
            if(u0Upper <0 || v0Upper<0 || u0Upper > 640 || v0Upper > 480 ){
                continue;
            }

            float u0fLower = match.pt.x - 1.0, v0fLower = - (a*u0fLower-c)/b;
            int u0Lower = (int)round(u0fLower), v0Lower = (int)round(v0fLower);
//            if(u0Lower <-640 || v0Lower<-640 || u0Lower > 640 || v0Lower > 640 ){
            if(u0Lower <0 || v0Lower<0 || u0Lower > 640 || v0Lower > 480 ){
                continue;
            }

            // derivate along epipolar line
            float g = (imageKf2.at<uchar>(u0Upper, v0Upper) - imageKf2.at<uchar>(u0Lower, v0Lower) ) / 2.0;
            float q = (gradientKf2.at<uchar>(u0Upper, v0Upper) - gradientKf2.at<uchar>(u0Lower, v0Lower) ) / 2.0;

            // intensity/gradient error
            float intensityError = intensity1 - match.intensity;
            float gradientError = gradient1 - match.gradient;

            // subpixel estimation of u0
            float u0Star = u0 + (g * intensityError + 1.0 / theta * q * gradientError ) / (g*g + 1.0/theta *q*q);
            float sigmaU0Star = sqrt( 2.0 * sigmaI * sigmaI /(g*g + 1.0/theta *q*q) );

            // inverse depth hypothese
            float rho = calInverseDepthEstimation(kp1, u0Star, pKF1, pKF2);
            if(isnan(rho)){
                continue;
            }

            float rhoUpper = calInverseDepthEstimation(kp1, u0Star + sigmaU0Star, pKF1, pKF2);
            float rhoLower = calInverseDepthEstimation(kp1, u0Star -  sigmaU0Star, pKF1, pKF2);
            float sigmaRho = max(abs(rhoUpper - rho),abs(rhoLower - rho));

            kp1.updateHypo(rho, sigmaRho);
            vMatchedIndices.push_back(make_pair(index1, matchedKeyPointIdx));
        }
    }
}

float MapReconstructor::calInverseDepthEstimation(RcKeyPoint &kp1,const float u0Star,KeyFrame *pKF1, KeyFrame *pKF2){

    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat rjiz = (cv::Mat_<float>(1,3) << R12.at<float>(2,0), R12.at<float>(2,1), R12.at<float>(2,2));
    cv::Mat rjix = (cv::Mat_<float>(1,3) << R12.at<float>(0,0), R12.at<float>(0,1), R12.at<float>(0,2));

    cv::Mat xp1 = (cv::Mat_<float>(1,3) << (kp1.pt.x-pKF1->cx)*pKF1->invfx, (kp1.pt.y-pKF1->cy)*pKF1->invfy, 1.0);

    float rho = (rjiz.dot(xp1) *(u0Star-pKF1->cx) - pKF1->fx * rjix.dot(xp1) ) / (-t12.at<float>(2) * (u0Star-pKF1->cx) + pKF1->fx * t12.at<float>(0) );

    return rho;
}

bool MapReconstructor::CheckDistEpipolarLine(RcKeyPoint &kp1,RcKeyPoint &kp2,cv::Mat &F12,KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

float MapReconstructor::calcInPlaneRotation(KeyFrame* pKF1,KeyFrame* pKF2){
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat R2w = pKF2->GetRotation();

    cv::Mat R12 = R1w*R2w.t();

    float roInPlane = fastAtan2(R12.at<uchar>(2,1), R12.at<uchar>(1,1));
    return roInPlane;
}

float MapReconstructor::calcMedianRotation(KeyFrame* pKF1,KeyFrame* pKF2)
{
    vector<float> angles;
    vector<KeyPoint> kp1s = pKF1->mvKeysUn;
    vector<KeyPoint> kp2s = pKF2->mvKeysUn;


    double median;
    size_t size = angles.size();

    sort(angles.begin(), angles.end());

    if (size  % 2 == 0)
    {
        median = (angles[size / 2 - 1] + angles[size / 2]) / 2;
    }
    else
    {
        median = angles[size / 2];
    }

    return median;
}

void MapReconstructor::StartKeyFrameQueueProcess()
{
	mStatus_KeyFrameQueueProcess = STARTED;
}

void MapReconstructor::StopKeyFrameQueueProcess()
{
	mStatus_KeyFrameQueueProcess = STOPPED;
}

void MapReconstructor::StartRealTimeMapReconstruction()
{
	mStatus_RealTimeMapReconstruction = STARTED;
}

void MapReconstructor::StopRealTimeMapReconstruction()
{
	mStatus_RealTimeMapReconstruction = STOPPED;
}

void MapReconstructor::StartFullMapReconstruction()
{
	mStatus_FullMapReconstruction = STARTED;
}

void MapReconstructor::StopFullMapReconstruction()
{
	mStatus_FullMapReconstruction = STOPPED;
}

} //namespace ORB_SLAM2


