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
    //normalize(pKeyFrame->mRefImgGray, pKeyFrame->mRefImgGray, 0x00, 0xFF, NORM_MINMAX, CV_8U);

    // ? stor opt
    pKeyFrame->mRefImgGradient = modulo;

    highGradientAreaKeyPoints(modulo,orientation, pKeyFrame, lambdaG);
}

void MapReconstructor::highGradientAreaKeyPoints(Mat &gradient, Mat &orientation, KeyFrame *pKF, const float gradientThreshold){
    Mat image = pKF->mRefImgGray;
    Mat depths = pKF->mRefImgDepth;

//     vector<RcKeyPoint> keyPoints;
     map<Point2f,RcKeyPoint,Point2fLess> keyPoints;

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

    for(int row = 0; row < gradient.rows; ++row) {
        uchar* p = gradient.ptr<uchar>(row);
        for(int col = 0; col < gradient.cols; ++col) {
            int gradientModulo = p[col];

            // 1. filter with intensity threshold
            if(gradientModulo <= gradientThreshold)
            {
                continue;
            }

            // 2. estimate octave
            // judge which interval of one point on edge belongs to, give an approximate estimation of octave
            int octave;
            float depth = depths.at<float>(Point(col, row));
//            float depth = depths.at<float>(row, col);
            for(iter = octaveDepthMap.end(); iter != octaveDepthMap.begin(); iter--)
            {
                octave = iter->first;
                if(depth > iter->second.second && depth<=iter->second.second){
                    break;
                }
            }
            //
            float angle = orientation.at<float>(Point(col, row));
            float intensity = image.at<float>(Point(col, row));
//            float angle = orientation.at<float>(row, col);
//            float intensity = image.at<float>(row, col);
            float gradient = gradientModulo;
            Point2f cord = Point2f(col, row);
//            Point2f cord = Point2f((float) row, (float) col);
            keyPoints[cord] = RcKeyPoint(col, row,intensity,gradient,angle,octave,depth);
//            keyPoints.insert(make_pair(cord, RcKeyPoint(col, row,intensity,gradient,angle,octave,depth)));
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
    realTimeReconstructionEnd = false;

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
        if (frameValid || retryCount>7)
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
//            sleep(3);
        }

        if(frameValid) {
            CreateNewMapPoints(currentKeyFrame);

            fuseHypo(currentKeyFrame);

//            {
//                unique_lock<mutex> lock(mMutexForKFQueueForReonstruction);
//                for(int i=0;i<2;i++)
//                {
//                    if(mlpKFQueueForReonstruction.empty())
//                    {
//                        break;
//                    }
//                    mlpKFQueueForReonstruction.pop_front();
//                }
//            }
        }
    }

    realTimeReconstructionEnd = true;

	cout << "MapReconstructor: End thread execution for map reconstruction during SLAM tracking." << endl;

	cout << "MapReconstructor: Start thread execution for full map reconstruction." << endl;

	// TODO: Remove the sleep process, once the real code is implemented.
//	usleep(10000);

	cout << "MapReconstructor: End thread execution for full map reconstruction." << endl;
}

bool MapReconstructor::isRealTimeReconstructionEnd()
{
    return realTimeReconstructionEnd;
}

bool MapReconstructor::CheckNewKeyFrames(KeyFrame* currentKeyFrame)
{
    const vector<KeyFrame*> vpNeighKFs = currentKeyFrame->GetBestCovisibilityKeyFrames(kN);
    return (int)vpNeighKFs.size() >= kN;
}

void MapReconstructor::CreateNewMapPoints(KeyFrame* mpCurrentKeyFrame)
{
    cout<<"CreateNewMapPoints"<<endl;
    // Retrieve neighbor keyframes in covisibility graph
    int nn = kN;
//    if(mbMonocular)
//        nn=2 * kN;

    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
//        if(i>0 && CheckNewKeyFrames())
//            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        long mnid2 = pKF2->mnId;
        // while ? TODO
        if(!keyframeKeyPointsMap.count(mnid2)){
            cout << "keyframe data not extracted yet: " << mnid2 << endl;
            continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);
        epipolarConstraientSearch(mpCurrentKeyFrame, pKF2, F12,vMatchedIndices);
    }
}

cv::Mat MapReconstructor::UnprojectStereo(RcKeyPoint &p,KeyFrame *pKF)
{
    const float z = p.mDepth;

//    vector<pair<float,float>> hypos = p.hypotheses;
//    float tho = 0;
//    for(size_t i=0;i<hypos.size();i++){
//        tho += hypos[i].first;
//    }
//    tho /= hypos.size();
//    if(tho==0){
//        return cv::Mat();
//    }
//        const float z = 1.0/tho;

    const float u = p.pt.x;
    const float v = p.pt.y;
//    return pKF->UnprojectStereo(u,v,z);
//    cout<<"new point , mDepth: " <<  p.mDepth<<", tho: "<<tho<<", invtho: "<<z<<endl;
    return pKF->UnprojectStereo(u,v,z);
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



bool MapReconstructor::getSearchAreaForWorld3DPointInKF ( KeyFrame* const  pKF1, KeyFrame* const pKF2, const RcKeyPoint& twoDPoint,int& lowerBoundXInKF2, int& lowerBoundYInKF2, int& upperBoundXInKF2, int& upperBoundYInKF2 )
{
    //Uproject lower and upper point from KF1 to world
    const float z = twoDPoint.mDepth;
    cv::Mat lower3Dw = cv::Mat();
     cv::Mat upper3Dw = cv::Mat();
    if(z>1000 && z<8000)
    {
        float lowerZ = 0.95*z;
        float upperZ = 1.05*z;
        const float u = twoDPoint.pt.x;
        const float v = twoDPoint.pt.y;
       //todo:  not use KeyFrame Version as it need to lock mutex
       lower3Dw = pKF1->UnprojectStereo(u,v, lowerZ);
        

        
        const float x = (u-pKF1->cx)*upperZ*pKF1->invfx;
        const float y = (v-pKF1->cy)*upperZ*pKF1->invfy;
        //todo:  not use KeyFrame Version as it need to lock mutex
        upper3Dw = pKF1->UnprojectStereo(u,v, upperZ);
        
    }
    else
        return false;
    
    //Project to  Neighbor KeyFrames
    
    float upperU = 0.0;
    float upperV = 0.0;
    float lowerU = 0.0;
     float lowerV = 0.0;
    bool isUpperValid = pKF2->ProjectStereo(upper3Dw, upperU, upperV);
   
 if(!isUpperValid) 
         return false;
    bool isLowerValid = pKF2->ProjectStereo(lower3Dw, lowerU, lowerV);
      
      if(!isLowerValid) 
         return false;
         
    if (lowerU > upperU)
        swap(lowerU, upperU);
    if(lowerV > upperV)
        swap(lowerV, upperV);
   
    return true;
    
}
void MapReconstructor::epipolarConstraientSearch(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,vector<pair<size_t,size_t> > &vMatchedIndices)
{
    cout<<"epipolarConstraientSearch"<<endl;
    const long mnid1 = pKF1->mnId, mnid2 = pKF2->mnId;
//    vMatchedIndices.clear();
    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints1 = keyframeKeyPointsMap.at(mnid1);
    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints2 = keyframeKeyPointsMap.at(mnid2);

//    float medianRotation = calcInPlaneRotation(pKF1,pKF2);
    float medianRotation = calcMedianRotation(pKF1,pKF2);
//    float tho0, sigma0;
//    calcSenceInverseDepthBounds(pKF1, tho0, sigma0);
//    cout<<"median dep"<< tho0<<" "<<sigma0<<" mro "<<medianRotation<<endl;

//    const float width = pKF2->mRefImgGray.rows, height = pKF2->mRefImgGray.cols;
    const float width = pKF2->mRefImgGray.cols, height = pKF2->mRefImgGray.rows;
    if(width ==0 || height ==0)
    {
        cout<<"invalid width height"<<endl;
        return;
    }

    for(auto &kpit : keyPoints1)
    {
        RcKeyPoint &kp1 = kpit.second;

        if(kp1.fused)
        {
//            cout<<"kp1 already fused"<<endl;
            continue;
        }

        // prepare data
        float intensity1 = kp1.intensity;
        float angle1 = kp1.orientation;
        float gradient1 = kp1.gradient;

        float minSimilarityError = -1.0;
        Point2f matchedCord;

        // epipolar line params
        const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
        const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
        const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

        if(a==0&&b==0)
        {
            continue;
        }

      //Reduce the search area in KF2 for corresponding high gradient point kp1
     
      int lowerBoundXInKF2, lowerBoundYInKF2, upperBoundXInKF2, upperBoundYInKF2;
      getSearchAreaForWorld3DPointInKF ( pKF1, pKF2, kp1, lowerBoundXInKF2, lowerBoundYInKF2, upperBoundXInKF2, upperBoundYInKF2 );
      
      
        /*
//        cout<<"move on epipolar line"<<endl;
        Point2f startCord,endCord;
        bool bounds = calCordBounds(startCord, endCord, width, height, a, b, c);
//        cout<<"end cal bounds"<<endl;
        if(!bounds)
        {
            continue;
        }
//        cout<<startCord<<endCord<<endl;
        // move on epipolar line
        float increaseRate = (endCord.y - startCord.y) / (endCord.x - startCord.x);
        float axisCrossP = startCord.y - increaseRate * startCord.x;
        if(isnan(increaseRate))
        {
            continue;
        }
        cout<<"increaseRate "<<increaseRate<<endl;

        Mat alreadyProcessed = Mat(height, width, CV_8U, Scalar::all(0));
        vector<Point> neighbourP;
        while(cordInImageBounds(startCord.x,startCord.y,width,height))
        {
            // create p 4 neighbours
            float x = startCord.x, y = startCord.y;
//            cout<<"cords "<<x<<", "<<y<<endl;
            neighbourP.clear();
            neighbourP.push_back(Point(floor(x), floor(y)));
            neighbourP.push_back(Point(ceil(x), floor(y)));
            neighbourP.push_back(Point(ceil(x), ceil(y)));
            neighbourP.push_back(Point(floor(x), ceil(y)));

            for(size_t ni=0;ni<neighbourP.size();ni++)
            {
                Point cord = neighbourP[ni];
                if(!cordInImageBounds(cord.x,cord.y,width,height))
                {
                    continue;
                }
                Point2f cordP = Point2f(cord.x, cord.y);
                if(alreadyProcessed.at<uchar>(cord) < 1)
                {
                    alreadyProcessed.at<uchar>(cord) = 1;

                    if(!keyPoints2.count(cordP))
                    {
                        continue;
                    }
                    RcKeyPoint &kp2 = keyPoints2.at(cordP);
                    if(kp2.fused)
                    {
                        continue;
                    }
                    float similarityError = checkEpipolarLineConstraient(kp1, kp2, a, b, c ,medianRotation,pKF2);
//                    cout<<"similarityError "<<similarityError<<endl;

                    // update the best match point
                    if((minSimilarityError < 0 && similarityError >=0) || minSimilarityError > similarityError){
                        minSimilarityError = similarityError;
                        matchedCord = cordP;
                    }
                }
            }
            float step = increaseRate < 1 ? 1.0 / increaseRate : 1.0;
            startCord = Point2f(x + step, increaseRate * (x + step) + axisCrossP);
        }
*/
        for(auto &kpit2 : keyPoints2)
        {
            RcKeyPoint &kp2 = kpit2.second;
            Point2f p2 = kpit2.first;

            if(kp2.fused){
//                cout<<"kp1 already fused"<<endl;
                continue;
            }

            // prepare data
            float intensity2 = kp2.intensity;
            float angle2 = kp2.orientation;
            float gradient2 = kp2.gradient;

            // search on epipolar line
//            if(!CheckDistEpipolarLine(kp1,kp2,F12,pKF2)){
//                continue;
//            }
            const float num = a*kp2.pt.x+b*kp2.pt.y+c;

            const float den = a*a+b*b;

            const float dsqr = num*num/den;

            if(!(dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave]))
            {
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
                matchedCord = Point2f(p2.x, p2.y);
            }
        }

//        cout<<"found "<<minSimilarityError<<endl;
        // use the best match point to estimate the distribution
        if(minSimilarityError >=0 && minSimilarityError<1.0e+5){
//            cout<<matchedCord<<"matchedCord "<<minSimilarityError<<endl;
            if(!keyPoints2.count(matchedCord)){
//                cout<<"can't find back"<<endl;
                continue;
            }

            RcKeyPoint &match = keyPoints2.at(matchedCord);

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
            if(!cordInImageBounds(u0Upper, v0Upper, width, height))
            {
                continue;
            }

            float u0fLower = match.pt.x - 1.0, v0fLower = - (a*u0fLower-c)/b;
            int u0Lower = (int)round(u0fLower), v0Lower = (int)round(v0fLower);
            if(!cordInImageBounds(u0Lower, v0Lower, width, height))
            {
                continue;
            }

            // derivate along epipolar line
            float g = (imageKf2.at<float>(Point(u0Upper, v0Upper)) - imageKf2.at<float>(Point(u0Lower, v0Lower) )) / 2.0;
            float q = (gradientKf2.at<float>(Point(u0Upper, v0Upper)) - gradientKf2.at<float>(Point(u0Lower, v0Lower)) ) / 2.0;

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

//            cout<<"add hypo"<<(1.0/rho)<<endl;
            kp1.addHypo(rho, sigmaRho,&match);
        }
    }
}

bool MapReconstructor::calCordBounds(Point2f &startCordRef, Point2f &endCordRef, float width, float height, float a, float b,float c)
{
    vector<Point2f> cords;

    bool xcord = false, ycord = false;
    if((a*a + b*b) ==0)
    {
        return false;
    }
    else if(a==0)
    {
        ycord = true;
    }
    else if(b==0)
    {
        xcord = true;
    }
    else
    {
        xcord = ycord = true;
    }

    if(ycord)
    {
//        cout<<"c "<<c<<", b "<<b<<endl;
        cords.push_back(Point2f(0.0, -c /b));
        cords.push_back(Point2f(width, (-c - a*width) /b));
    }
    if(xcord)
    {
//        cout<<"c "<<c<<", a "<<a<<endl;
        cords.push_back(Point2f(-c / a, 0.0));
        cords.push_back(Point2f((-c - b*height) / a, height));
    }
//    cout<<"push end "<<cords<<endl;

    int cordSize = (int)cords.size();
    if(cordSize < 2)
    {
        return false;
    }

    vector<int> validIndexes;
    int validCount=0;
    for(int i = 0; i < (int)cords.size();++i)
    {
        Point2f p = cords.at(i);
        const float xc = p.x;
        const float yc = p.y;
        if(!isnan(xc) && !isnan(yc) && cordInImageBounds(xc,yc,width,height))
        {
            validIndexes.push_back(i);
            validCount ++;
        }
//        cout<<"i "<<i<<" p "<<p<<" valid "<<cordInImageBounds(xc,yc,width,height)<<"w h"<<width<<height<<endl;
    }
//    vector<Point2f>::iterator cordit;
//    cout<<"siz "<<cordSize<<endl;
//    for(cordit=cords.begin();cordit!=cords.end();++cordit)
//    {
//        const float xc = cordit->x;
//        const float yc = cordit->y;
////        cout<<"erase "<<xc<<" "<<yc<<endl;
//        if(isnan(xc) || isnan(yc) || !cordInImageBounds(xc,yc,width,height))
//        {
//            cords.erase(cordit);
//        }
////        cout<<"erase end"<<endl;
//    }
    if(validCount < 2)
    {
//        cout<<"invalid cord ed "<<endl;
        return false;
    }
//    cout<<"validIndexes "<<validIndexes.size()<<endl;

    int startIndex = 0, endIndex = 1;
    Point2f startCord, endCord;
    startCord = cords[validIndexes[startIndex]];
    while(endIndex<validCount)
    {
        endCord = cords[validIndexes[endIndex]];
        if(endCord.x != startCord.x || endCord.y != startCord.y )
        {
            break;
        }
        endIndex++;
    }

//    cout<<"cord "<<startCord<<", "<<endCord<<endl;

    if(endCord.x - startCord.x <0)
    {
        //exchange
        startCord = cords[validIndexes[endIndex]];
        endCord = cords[validIndexes[startIndex]];
    }
//    cout<<"cord "<<startCord<<", "<<endCord<<endl;

    startCordRef.x = startCord.x;
    startCordRef.y = startCord.y;
    endCordRef.x = endCord.x;
    endCordRef.y = endCord.y;
    return true;
}

float MapReconstructor::checkEpipolarLineConstraient(RcKeyPoint &kp1, RcKeyPoint &kp2, float a, float b, float c, float medianRotation, KeyFrame *pKF2)
{
    float similarityError = -1.0;
    // prepare data
    float intensity1 = kp1.intensity;
    float angle1 = kp1.orientation;
    float gradient1 = kp1.gradient;

    float intensity2 = kp2.intensity;
    float angle2 = kp2.orientation;
    float gradient2 = kp2.gradient;

//    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

//    const float den = a*a+b*b;

//    if(den==0)
//    {
//        return similarityError;
//    }

//    const float dsqr = num*num/den;

//    if(dsqr > 3.84*pKF2->mvLevelSigma2[kp2.octave])
//    {
//        return similarityError;
//    }

    // check epipolar line angle with orientation
    float eplAngle = fastAtan2(-b,a);
    float eplAngleDiff  = angle2 - eplAngle ;
    if(eplAngleDiff < -0.0){
        eplAngleDiff += 360.0;
    }else if(eplAngleDiff > 360.0){
        eplAngleDiff -= 360.0;
    }
    if(eplAngleDiff >= lambdaL){
        return similarityError;
    }

    // check in-plane rotation
//    float medianRotation = calcInPlaneRotation(pKF1,pKF2);
    float angleDiff = angle2 - angle1 - medianRotation;
    if(angleDiff < -0.0){
        angleDiff += 360.0;
    }else if(angleDiff > 360.0){
        angleDiff -= 360.0;
    }
    if(angleDiff >= lambdaThe){
        return similarityError;
    }

    // cal similarity error
    float intensityError = intensity1 - intensity2;
    float gradientError = gradient1 - gradient2;
    similarityError = (intensityError * intensityError + 1.0/theta * gradientError * gradientError ) / ( sigmaI * sigmaI );
    return similarityError;
}

float MapReconstructor::calInverseDepthEstimation(RcKeyPoint &kp1,const float u0Star,KeyFrame *pKF1, KeyFrame *pKF2){

    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

//    cv::Mat R12 = R1w*R2w.t();
    cv::Mat R21 = R2w*R1w.t();
//    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
    cv::Mat t21 = -R2w*R1w.t()*t1w+t2w;

//    cv::Mat rjiz = (cv::Mat_<float>(1,3) << R12.at<float>(2,0), R12.at<float>(2,1), R12.at<float>(2,2));
//    cv::Mat rjix = (cv::Mat_<float>(1,3) << R12.at<float>(0,0), R12.at<float>(0,1), R12.at<float>(0,2));
    cv::Mat rjiz = (cv::Mat_<float>(1,3) << R21.at<float>(2,0), R21.at<float>(2,1), R21.at<float>(2,2));
    cv::Mat rjix = (cv::Mat_<float>(1,3) << R21.at<float>(0,0), R21.at<float>(0,1), R21.at<float>(0,2));

    cv::Mat xp1 = (cv::Mat_<float>(1,3) << (kp1.pt.x-pKF1->cx)*pKF1->invfx, (kp1.pt.y-pKF1->cy)*pKF1->invfy, 1.0);

//    float rho = (rjiz.dot(xp1) *(u0Star-pKF1->cx) - pKF1->fx * rjix.dot(xp1) ) / (-t12.at<float>(2) * (u0Star-pKF1->cx) + pKF1->fx * t12.at<float>(0) );
    float rho = (rjiz.dot(xp1) *(u0Star-pKF1->cx) - pKF1->fx * rjix.dot(xp1) ) / (-t21.at<float>(2) * (u0Star-pKF1->cx) + pKF1->fx * t21.at<float>(0) );

    return rho;
}

bool MapReconstructor::CheckDistEpipolarLine(RcKeyPoint &kp1,RcKeyPoint &kp2,cv::Mat &F12,KeyFrame* pKF)
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

    return dsqr<3.84*pKF->mvLevelSigma2[kp2.octave];
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

    const map<Point2f,RcKeyPoint,Point2fLess> &keyPoints1 = keyframeKeyPointsMap.at(pKF1->mnId);
    const map<Point2f,RcKeyPoint,Point2fLess> &keyPoints2 = keyframeKeyPointsMap.at(pKF2->mnId);

    const std::vector<cv::KeyPoint> mvKeys1 = pKF1->mvKeys;
    const std::vector<cv::KeyPoint> mvKeys2 = pKF2->mvKeys;

    vector<float> angles;
    float median = 0.0;

//    vector<MapPoint*> mMps = pKF1->GetMapPointMatches();
    set<MapPoint*> mMps = pKF1->GetMapPoints();
    for(set<MapPoint*>::iterator vitMP=mMps.begin(), vendMP=mMps.end(); vitMP!=vendMP; vitMP++)
    {
        MapPoint* mp = *vitMP;
        if(!mp)
            continue;
        if(mp->isBad())
            continue;

        if(mp->IsInKeyFrame(pKF2))
        {
            int idx1 = mp->GetIndexInKeyFrame(pKF1);
            int idx2 = mp->GetIndexInKeyFrame(pKF2);

            KeyPoint kp1 = mvKeys1[idx1];
            KeyPoint kp2 = mvKeys2[idx2];
            Point2f c1 = Point2f(kp1.pt.x, kp1.pt.y);
            Point2f c2 = Point2f(kp2.pt.x, kp2.pt.y);

//            cout<<keyPoints1.count(c1)<<","<<keyPoints2.count(c2)<<endl;
            if(keyPoints1.count(c1) && keyPoints2.count(c2))
            {
                float ort2 = keyPoints2.at(c2).orientation, ort1 = keyPoints1.at(c1).orientation;
                angles.push_back(ort2 - ort1);
            }
        }
    }
    if(angles.empty())
    {
        return median;
    }

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

void MapReconstructor::calcSenceInverseDepthBounds(KeyFrame* pKF, float &tho0, float &sigma0)
{
    const std::vector<float> mvDepth = pKF->mvDepth;
    float maxDepth = NAN, minDepth = NAN;
    for (size_t i = 0; i < mvDepth.size(); ++i)
    {
        float depth = mvDepth[i];
        if(isnan(maxDepth))
        {
            maxDepth = minDepth = depth;
            continue;
        }
        else
        {
            if(depth > maxDepth)
            {
                maxDepth = depth;
            }
            else if(depth < minDepth)
            {
                minDepth = depth;
            }
        }
    }
    tho0 = 1.0 / minDepth;
    minDepth = 1.0 / maxDepth;
    maxDepth = tho0;

    tho0 = (minDepth + maxDepth) / 2.0;
    sigma0 = (maxDepth - minDepth) / 4.0;
}

bool MapReconstructor::cordInImageBounds(float x, float y, int width, int height)
{
    return (x>=0.0 && x<=width && y>=0.0 && y<=height);
}



void MapReconstructor::fuseHypo(KeyFrame* pKF)
{
    cout<< "enter fuse"<<endl;
    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints = keyframeKeyPointsMap.at(pKF->mnId);
    for(auto &kpit : keyPoints)
    {
        RcKeyPoint &kp1 = kpit.second;
        if(kp1.fused)
        {
//            cout<<"already fs"<<endl;
            continue;
        }
        //        cout<<kp1.hypotheses.size()<<endl;
        vector<pair<float, float>> hypos = kp1.hypotheses;
        int total = hypos.size();
        if(total==0)
        {
//            continue;
        }
        if(total > lambdaN)
        {
            float sunTho = 0, sunSigmaInv = 0;
            set<int> nearest;
            // ka square check between every two hypotheses
            for (int i = 0; i < total; i++)
            {
                for (int j = i + 1; j < total; j++)
                {
                    pair<float, float> a = hypos[i], b = hypos[j];
                    float diffSqa = (a.first - b.first) * (a.first - b.first);
                    float sigmaSqi = a.second * a.second, sigmaSqj = b.second*b.second;
                    if((diffSqa/sigmaSqi+ diffSqa/sigmaSqj) < 5.99)
                    {
                        if(!nearest.count(i))
                        {
                            sunTho += (a.first/sigmaSqi);
                            sunSigmaInv += (1.0/sigmaSqi);
                            nearest.insert(i);
                            //                            cout<<"insert "<<i<<endl;
                        }
                        if(!nearest.count(j))
                        {
                            sunTho += (b.first/sigmaSqj);
                            sunSigmaInv += (1.0/sigmaSqj);
                            nearest.insert(j);
                            //                            cout<<"insert "<<j<<endl;
                        }
                    }
                }
            }
            if((int)nearest.size() > lambdaN)
            {
                // fuse
                float tho = sunTho / sunSigmaInv;
                float sigmaSquare = 1.0 / sunSigmaInv;
float thoM = kp1.mDepth;
                kp1.fused = true;
                kp1.tho = tho;
                kp1.sigma = sqrt(sigmaSquare);

                // set hypotheses fuse flag
                map<RcKeyPoint*,int> &rel = kp1.hypothesesRelation;
int hypoCount = 1;
                for(auto &kpit2 : rel)
                {
                    kpit2.first->fused = true;
                    cout<<"kpit2 fused true"<<endl;
thoM += kpit2.first->mDepth;
hypoCount++;
                }
kp1.mDepth = thoM / hypoCount;
            }
        }


//        cout<<"proj"<<endl;
        cv::Mat x3D = UnprojectStereo(kp1, pKF);
        if(x3D.empty()){
            continue;
        }
//        cout<<"proj succ"<<endl;

        // Triangulation is succesfull
        MapPoint* pMP = new MapPoint(x3D,pKF,mpMap);

//            pMP->AddObservation(mpCurrentKeyFrame,idx1);
//            pMP->AddObservation(pKF2,idx2);

//            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
//            pKF2->AddMapPoint(pMP,idx2);

//            pMP->ComputeDistinctiveDescriptors();

//            pMP->UpdateNormalAndDepth();

        mpMap->AddMapPoint(pMP);
//            mlpRecentAddedMapPoints.push_back(pMP);
    }
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


