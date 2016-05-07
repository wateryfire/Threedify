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
    //todo: Exception handling and default value
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    kN = fSettings["ReConstruction.KN"];
    sigmaI = fSettings["ReConstruction.sigmaI"];
    lambdaG = fSettings["ReConstruction.lambdaG"];
    lambdaL = fSettings["ReConstruction.lambdaL"];
    lambdaThe = fSettings["ReConstruction.lambdaThe"];
    lambdaN = fSettings["ReConstruction.lambdaN"];
    theta = fSettings["ReConstruction.theta"];

    // camera params
    width = fSettings["Camera.width"];
    height = fSettings["Camera.height"];

    //DEBUG
    depthThresholdMax = fSettings["ReConstruction.maxDepth"];
    depthThresholdMin = fSettings["ReConstruction.minDepth"];
    epipolarSearchOffset = fSettings["ReConstruction.searchOffset"];
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

        currentKeyFrame->mRefImgGray.refcount = 0;
        currentKeyFrame->mRefImgGray.release();
        currentKeyFrame->mRefImgDepth.refcount = 0;
        currentKeyFrame->mRefImgDepth.release();

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
//    normalize(pKeyFrame->mRefImgGray, pKeyFrame->mRefImgGray, 0x00, 0xFF, NORM_MINMAX, CV_8U);

    highGradientAreaKeyPoints(modulo,orientation, pKeyFrame, lambdaG);
}

void MapReconstructor::highGradientAreaKeyPoints(Mat &gradient, Mat &orientation, KeyFrame *pKF, const float gradientThreshold){
    Mat image = pKF->mRefImgGray;
    Mat depths = pKF->mRefImgDepth;

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

            if(!pKF->IsInImage(col,row))continue;

            float depth = depths.at<float>(Point(col, row));
            if(depth<=depthThresholdMin || depth > depthThresholdMax)
            {
                continue;
            }

            // 2. estimate octave
            // judge which interval of one point on edge belongs to, give an approximate estimation of octave

            int octave;
            for(iter = octaveDepthMap.end(); iter != octaveDepthMap.begin(); iter--)
            {
                octave = iter->first;
                if(depth > iter->second.first && depth<=iter->second.second){
                    break;
                }
            }
            //
            float angle = orientation.at<float>(Point(col, row));
            float intensity = image.at<float>(Point(col, row));
            Point2f cord = Point2f(col, row);
            RcKeyPoint hgkp(col, row,intensity,gradientModulo,angle,octave,depth);

            // fill neighbour info
            hgkp.fetchNeighbours(image, gradient);

            // undistort
//            cv::Mat mat(1,2,CV_32F);
//            mat.at<float>(0,0)=hgkp.pt.x;
//            mat.at<float>(0,1)=hgkp.pt.y;

//            mat=mat.reshape(2);
//            cv::undistortPoints(mat,mat,pKF->mK,mDistCoef,cv::Mat(), pKF->mK);
//            mat=mat.reshape(1);

//            hgkp.pt.x=mat.at<float>(0,0);
//            hgkp.pt.y=mat.at<float>(0,1);

            keyPoints[cord] = hgkp;
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

    std::deque<KeyFrame*> interKeyFrameCheckingStack;
    KeyFrame* currentKeyFrameInterChecking=NULL;

    KeyFrame* currentKeyFrame=NULL;

	while(mStatus_RealTimeMapReconstruction!=STARTED)
    {
        sleep(1);
	}

	cout << "MapReconstructor: Start thread execution for map reconstruction during SLAM tracking." << endl;

    int retryCount = 0;
    while(mStatus_RealTimeMapReconstruction!=STOPPED)
    {
        if(mlpKFQueueForReonstruction.empty())
        {
            sleep(1);

            continue;
        }

        currentKeyFrame = mlpKFQueueForReonstruction.front();
        cout << "MapReconstructor: Reconstructing map from the key frame (FrameId: " << currentKeyFrame->mnId << ")." << endl;

        bool frameValid = CheckNewKeyFrames(currentKeyFrame);
        if (frameValid /*|| retryCount>7*/)
        {
            //Get Mutex-lock to access the queue of key frames.
            {
                unique_lock<mutex> lock(mMutexForKFQueueForReonstruction);
                mlpKFQueueForReonstruction.pop_front();
                interKeyFrameCheckingStack.push_back(currentKeyFrame);
            }
            retryCount=0;
        }
        else
        {
            retryCount ++;
            sleep(1);
            continue;
//            sleep(3);
        }

        if(frameValid) {

//            processedKeyFrame = mlpKFQueueForReonstruction.;
            CreateNewMapPoints(currentKeyFrame);

            fuseHypo(currentKeyFrame);

            intraKeyFrameChecking(currentKeyFrame);



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

        if((int)interKeyFrameCheckingStack.size() > kN)
        {
            currentKeyFrameInterChecking = interKeyFrameCheckingStack.front();
            interKeyFrameChecking(currentKeyFrameInterChecking);
            interKeyFrameCheckingStack.pop_front();
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
//    const vector<KeyFrame*> vpNeighKFs = currentKeyFrame->GetBestCovisibilityKeyFrames(kN);
//    return (int)vpNeighKFs.size() >= kN;
    return (int)mlpKFQueueForReonstruction.size() > 10;
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
    float z = p.mDepth;

//    vector<pair<float,float>> hypos = p.hypotheses;
//    float tho = 0;
//    for(size_t i=0;i<hypos.size();i++){
//        tho += hypos[i].first;
//    }
//    tho /= hypos.size();
//    if(tho==0){
//        return cv::Mat();
//    }
//        const float zh = 1.0/tho;
//        cout<<"hypo zh "<<zh<<", measured "<<z<<endl;


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

void MapReconstructor::epipolarConstraientSearch(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,vector<pair<size_t,size_t> > &vMatchedIndices)
{
    // get rotation (j - i) and
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R21 = R2w*R1w.t();
    cv::Mat t21 = -R2w*R1w.t()*t1w+t2w;

    cv::Mat rjiz = (cv::Mat_<float>(1,3) << R21.at<float>(2,0), R21.at<float>(2,1), R21.at<float>(2,2));
    cv::Mat rjix = (cv::Mat_<float>(1,3) << R21.at<float>(0,0), R21.at<float>(0,1), R21.at<float>(0,2));
    float tjiz = t21.at<float>(2);
    float tjix = t21.at<float>(0);

    // high gradient area points
    const long mnid1 = pKF1->mnId, mnid2 = pKF2->mnId;
    cout<<"epipcolar constraient search between "<<mnid1<<" "<<mnid2<<endl;
//    vMatchedIndices.clear();
    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints1 = keyframeKeyPointsMap.at(mnid1);
    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints2 = keyframeKeyPointsMap.at(mnid2);

//    float medianRotation = calcInPlaneRotation(pKF1,pKF2);
    float medianRotation = calcMedianRotation(pKF1,pKF2);

    // inverse depth of sense
    float tho0, sigma0;
    calcSenceInverseDepthBounds(pKF1, tho0, sigma0);
    //sigma0 /= 2; //DEBUG
    cout<<"median dep"<< tho0<<" "<<sigma0<<" mro "<<medianRotation<<endl;

    // search for each point in first image
    for(auto &kpit : keyPoints1)
    {
        RcKeyPoint &kp1 = kpit.second;

        // prepare data
        float intensity1 = kp1.intensity;
//        float angle1 = kp1.orientation;
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

        // Xba(p) = K.inv() * xp
        cv::Mat xp1 = (cv::Mat_<float>(1,3) << (kp1.pt.x-pKF1->cx)*pKF1->invfx, (kp1.pt.y-pKF1->cy)*pKF1->invfy, 1.0);
//        cv::Mat xp = (cv::Mat_<float>(3,1) << kp1.pt.x, kp1.pt.y, 1.0);
//        cv::Mat xp1 = pKF1->mK.inv() * xp;

        ////////////////////////
        /// fix the search area nearby the projection
        //tho0 = 1/kp1.mDepth;
        ////////////////////////

        float thoMax = tho0 + 2*sigma0, thoMin = tho0 - 2*sigma0;
        float u0 = pKF1->cx + (rjix.dot(xp1) + thoMax * tjix) / (rjiz.dot(xp1) + thoMax*tjiz) * pKF1->fx;
        float u1 = pKF1->cx + (rjix.dot(xp1) + thoMin * tjix) / (rjiz.dot(xp1) + thoMin*tjiz) * pKF1->fx;

//        float up = pKF1->cx + (rjix.dot(xp1) + tho0 * tjix) / (rjiz.dot(xp1) + tho0*tjiz) * pKF1->fx;

        ////////////////////////
        /// \brief minU
        ///
//        int lowerBoundXInKF2, lowerBoundYInKF2, upperBoundXInKF2, upperBoundYInKF2;
//        bool valid = getSearchAreaForWorld3DPointInKF( pKF1, pKF2, kp1,lowerBoundXInKF2, lowerBoundYInKF2, upperBoundXInKF2, upperBoundYInKF2 );
//        lowerBoundXInKF2 /=2;
//        lowerBoundYInKF2/=2;
//        upperBoundXInKF2 /=2;
//        upperBoundYInKF2 /=2;
//        if(!valid) continue;
//        u0=lowerBoundXInKF2;
//        u1=upperBoundXInKF2;
        ///
        /// ////////////////////

//        if(fabs(u0 - u1) < 0.1) continue;

        float minU = min(u0, u1), maxU = max(u0, u1);
        float minV = 0, maxV = 0;
//        cout<<"proj bound of u "<<minU<<", "<<maxU<<endl;

        float offset = epipolarSearchOffset, dx, dy;
        //////
        ///
//        minV = lowerBoundYInKF2;
//        maxV = upperBoundYInKF2;
//        offset = 0;
//        b=0;
        ///
        //////
        float offsetU = sqrt(offset * offset * a * a / (a*a + b*b));
        float offsetV = sqrt(offset * offset * b * b / (a*a + b*b));

        bool parralexWithYAxis = (b==0 || fabs(-a / b) > (float)(height/(2*(offset + 0.5f)));

        Point2f startCord;
        Point2f endCord;
        if(parralexWithYAxis)
        {
            minV = 0;
            maxV = height;
            //////
            ///
//            minV = lowerBoundYInKF2;
//            maxV = upperBoundYInKF2;
            ///
            //////
            startCord.x = minU;
            startCord.y = minV;
            dx = 1.0;
            dy = 0;
        }
        else
        {
            startCord.x = minU;
            startCord.y = -(c + a * minU) / b;
            minV = maxV = startCord.y;
            dx = 1.0;
            dy = -a / b;
        }
//        cout<<"init p "<<startCord<<endl;

        if(!cordInImageBounds(startCord.x,startCord.y,width,height))
        {
            bool bounds = calCordBounds(startCord, endCord, width, height, a, b, c);
            //cout<<"bounds p "<<startCord<<endCord<<endl;
            if(!bounds)
            {
//                cout<<"out of bound "<<endl;
                continue;
            }
            else
            {
                minU = max(startCord.x, minU);
                maxU = min(maxU, endCord.x);
                if(!parralexWithYAxis)
                {
                    minV = -(c + a * minU) / b;
                    maxV = -(c + a * minU) / b;
                }
            }
        }

        minU -= offsetU;
        maxU += offsetU;
        minV -= offsetV;
        maxV += offsetV;
        startCord.x = minU;
        startCord.y = minV;

//        cout<<"start "<<startCord<<endl;
//        cout<<"minU "<<minU<<endl;
//        cout<<"maxU "<<maxU<<endl;
//        cout<<"maxV "<<maxV<<endl;
//        cout<<"minV "<<minV<<endl;
//        cout<<"offsetU "<<offsetU<<endl;
//        cout<<"offsetV "<<offsetV<<endl;
//        cout<<"dx "<<dx<<endl;
//        cout<<"dy "<<dy<<endl;
//        cout<<"u0 "<<u0<<endl;
//        cout<<"u1 "<<u1<<endl;
//        cout<<"up "<<up<<endl;

        Point2f cordP;
        while(startCord.x < (maxU + 1.0))
        {
            float x = startCord.x, y = startCord.y;
//            cout<< "stx "<<startCord<<endl;
            while(y<(maxV + 1.0))
            {
               cordP.x = floor(x);
               cordP.y= floor(y);
//               cout<< "stP "<<cordP<<endl;
                if(keyPoints2.count(cordP))
                {
                    RcKeyPoint &kp2 = keyPoints2.at(cordP);
                    //if(!kp2.fused)
                    //{
                        float similarityError = checkEpipolarLineConstraient(kp1, kp2, a, b, c ,medianRotation,pKF2);

                        // update the best match point
                        if((minSimilarityError < 0 && similarityError >=0) || minSimilarityError > similarityError){
                            minSimilarityError = similarityError;
                            matchedCord = Point2f(cordP.x, cordP.y);
                        }
                    //}
                }

                y += 1.0;
            }

            startCord.x += dx;
            startCord.y += dy;
            if(!parralexWithYAxis)
            {
                maxV += dy;
            }
        }

        // use the best match point to estimate the distribution
        if(minSimilarityError >=0 && minSimilarityError<1.0e+3){
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
            float u0 = match.pt.x;

            // neighbour info of this point
            vector<float> upper,lower;
            match.getNeighbourAcrossLine(a,b,lower,upper);

            float intensityUpper = upper[0],gradientUpper = upper[1];
            float intensityLower = lower[0],gradientLower = lower[1];

            // derivate along epipolar line
            float g = (intensityUpper - intensityLower) / 2.0;
            float q = (gradientUpper - gradientLower) / 2.0;

            // intensity/gradient error
            float intensityError = intensity1 - match.intensity;
            float gradientError = gradient1 - match.gradient;

            // subpixel estimation of u0
            float errorSquare = (g*g + 1.0/theta *q*q);
            float leftPart = (g * intensityError + 1.0 / theta * q * gradientError ) / errorSquare;
            if(leftPart>1)
            {
//                cout<<"left part two large "<<leftPart<<", devide "<< errorSquare<<endl;
                leftPart = 0;
                errorSquare = max(2.0f, errorSquare);
            }
            float u0Star = u0 + leftPart;
            float sigmaU0Star = sqrt( 2.0 * sigmaI * sigmaI /errorSquare );

            // inverse depth hypothese
//            float rho = calInverseDepthEstimation(kp1, u0Star, pKF1, pKF2);
            float rho = (rjiz.dot(xp1) *(u0Star-pKF1->cx) - pKF1->fx * rjix.dot(xp1) ) / (-tjiz * (u0Star-pKF1->cx) + pKF1->fx * tjix );
            if(isnan(rho)){
                continue;
            }

//            float rhoUpper = calInverseDepthEstimation(kp1, u0Star + sigmaU0Star, pKF1, pKF2);
            float u0Starl = u0Star - sigmaU0Star, u0Starr = u0Star + sigmaU0Star;
            float rhoUpper = (rjiz.dot(xp1) *(u0Starr-pKF1->cx) - pKF1->fx * rjix.dot(xp1) ) / (-tjiz * (u0Starr-pKF1->cx) + pKF1->fx * tjix );
//            float rhoLower = calInverseDepthEstimation(kp1, u0Star -  sigmaU0Star, pKF1, pKF2);
            float rhoLower = (rjiz.dot(xp1) *(u0Starl-pKF1->cx) - pKF1->fx * rjix.dot(xp1) ) / (-tjiz * (u0Starl-pKF1->cx) + pKF1->fx * tjix );
            float sigmaRho = max(fabs(rhoUpper - rho),fabs(rhoLower - rho));

//            cout<<"add hypo"<<(1.0/rho)<<" xp* "<<u0Star<<" "<<v0Star<<endl;
            kp1.addHypo(rho, sigmaRho,&match);
        }
    }
}


bool MapReconstructor::getSearchAreaForWorld3DPointInKF ( KeyFrame* const  pKF1, KeyFrame* const pKF2, const RcKeyPoint& twoDPoint,int& lowerBoundXInKF2, int& lowerBoundYInKF2, int& upperBoundXInKF2, int& upperBoundYInKF2 )
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
       for ( int zindex = 0; zindex < 2; zindex ++)
       {
                cv::Mat P3Dc = (cv::Mat_<float>(3,1) <<XcEst, YcEst, ZcBound[zindex]);
                cv::Mat P3Dw=KF1Twc.rowRange(0,3).colRange(0,3)*P3Dc+KF1Twc.rowRange(0,3).col(3);
                boundPoints.push_back( P3Dw);
       }
         

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
//    cout <<"bound points"<<endl;
    for(auto & bp: boundPoints)
    {
         valid = pKF2->ProjectStereo(bp, tempU, tempV);
         if(!valid) 
                return false;
        
//         cout << tempU<< "  " <<       tempV << endl;
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
        cords.push_back(Point2f(0.0, -c /b));
        cords.push_back(Point2f(width, (-c - a*width) /b));
    }
    if(xcord)
    {
        cords.push_back(Point2f(-c / a, 0.0));
        cords.push_back(Point2f((-c - b*height) / a, height));
    }

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
    float eplAngle = fastAtan2(-a,b);
    float eplAngleDiff  = angle2 - eplAngle ;
        /*if(eplAngleDiff < 0){
            eplAngleDiff += 360.0;
        }else if(eplAngleDiff > 360){
            eplAngleDiff -= 360.0;
        }
eplAngleDiff = min(fabs(eplAngleDiff + 180), fabs(eplAngleDiff - 180));*/
    while(eplAngleDiff <0 || eplAngleDiff > 90)
    {
        if(eplAngleDiff < 0){
            eplAngleDiff += 360.0;
        }else if(eplAngleDiff > 360){
            eplAngleDiff -= 360.0;
        }
        else{
            if(eplAngleDiff>180){
                eplAngleDiff-=180.0;
            }else if(eplAngleDiff>90){
                eplAngleDiff=180.0 - eplAngleDiff;
            }
        }
    }
    if(eplAngleDiff >= lambdaL){
        return similarityError;
    }

    // check in-plane rotation
//    float medianRotation = calcInPlaneRotation(pKF1,pKF2);
    float angleDiff = angle2 - angle1 - medianRotation;
/*if(angleDiff < 0){
            angleDiff += 360.0;
        }else if(angleDiff > 360){
            angleDiff -= 360.0;
        }*/
angleDiff = fabs(angleDiff);

    /*while(angleDiff <0 || angleDiff > 90)
    {
        if(angleDiff < 0){
            angleDiff += 360.0;
        }else if(angleDiff > 360){
            angleDiff -= 360.0;
        }
        else{
            if(angleDiff>180){
                angleDiff-=180.0;
            }else if(angleDiff>90){
                angleDiff=180.0 - angleDiff;
            }
        }
    }*/

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

    cv::Mat R21 = R2w*R1w.t();

    float roInPlane = fastAtan2(R21.at<uchar>(2,1), R21.at<uchar>(1,1));
    return roInPlane;
}

float MapReconstructor::calcMedianRotation(KeyFrame* pKF1,KeyFrame* pKF2)
{

    const map<Point2f,RcKeyPoint,Point2fLess> &keyPoints1 = keyframeKeyPointsMap.at(pKF1->mnId);
    const map<Point2f,RcKeyPoint,Point2fLess> &keyPoints2 = keyframeKeyPointsMap.at(pKF2->mnId);

    const std::vector<cv::KeyPoint> &mvKeys1 = pKF1->mvKeys;
    const std::vector<cv::KeyPoint> &mvKeys2 = pKF2->mvKeys;

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
//float diff = ort2 - ort1;
//if(diff>360) diff-=360.0;
//else if(diff<0) diff+=360.0;
                angles.push_back(ort2-ort1);
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
        if(depth <= 0)
        {
            continue;
        }
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
    long kfid = pKF->mnId;
    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints = keyframeKeyPointsMap.at(kfid);
    for(auto &kpit : keyPoints)
    {
        RcKeyPoint &kp1 = kpit.second;
        //        cout<<kp1.hypotheses.size()<<endl;
        vector<pair<float, float>> &hypos = kp1.hypotheses;

        // DEBUG
//        hypos.push_back(make_pair(1/kp1.mDepth, kp1.mDepth * 0.01));

        set<int> nearest;
        int totalCompact = KaTestFuse(hypos, kp1.tho, kp1.sigma, nearest);

        if(totalCompact<=lambdaN)
        {
            continue;
        }

        kp1.fused = true;

        // set hypotheses fuse flag
//        map<RcKeyPoint*,int> &rel = kp1.hypothesesRelation;
//        for(auto &kpit2 : rel)
//        {
//            int fxidx = kpit2.second;
//            for(const int fsi : nearest)
//            {
//                if(fsi == fxidx)
//                {
//                    RcKeyPoint* pkp2 = kpit2.first;
//                    pkp2->fused = true;
//                }
//            }
//        }
    }
}

void MapReconstructor::intraKeyFrameChecking(KeyFrame* pKF)
{
    long kfid = pKF->mnId;
    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints = keyframeKeyPointsMap.at(kfid);
    for(auto &kpit : keyPoints)
    {
        RcKeyPoint &kp1 = kpit.second;
        //check neighbour hypos
        int neighbourHypos = 0;

        vector<pair<float, float>> nbrhypos;

        vector<int> matchedIndexes;
        int index = 0;
        kp1.eachNeighbourCords([&](Point2f pt){
            if(keyPoints.count(pt))
            {
                RcKeyPoint &kpn = keyPoints.at(pt);
                if(kpn.fused)
                {
                    nbrhypos.push_back(make_pair(kpn.tho, kpn.sigma));
                    neighbourHypos++;
                    matchedIndexes.push_back(index);
                }
            }

            index ++;
        });

        if(neighbourHypos<2)
        {
            continue;
        }
        // ka square test
        float tho, sigma;
        set<int> nearest;
        int totalCompact = KaTestFuse(nbrhypos, tho, sigma, nearest);
        if(totalCompact < 2)
        {
            // denoise without neighbour hypos support
            continue;
        }
        else
        {
            // grow with neighbour hypos support
            if(!kp1.fused)
            {
                // check index
                sort(matchedIndexes.begin(), matchedIndexes.end());
                if(abs(matchedIndexes.back() - matchedIndexes.front()) < 4) continue;
                kp1.fused = true;
                kp1.tho = tho;
                kp1.sigma = sigma;
            }

            kp1.intraCheckCount = totalCompact;
//            kp1.addHypo(tho, sigma, 0);
        }
    }
}

int MapReconstructor::KaTestFuse(vector<pair<float, float>> &hypos, float &tho, float &sigma, set<int> &nearest)
{
    int total = hypos.size();
    int fusedCount = 0;
    if(total != 0)
    {
        float sumTho = 0, sumSigmaInv = 0;
        // ka square check between every two hypotheses
        for (int i = 0; i < total; i++)
        {
            for (int j = i + 1; j < total; j++)
            {
                pair<float, float> a = hypos[i], b = hypos[j];

                float diffSqa = (a.first - b.first) * (a.first - b.first);
                float sigmaSqi = a.second * a.second, sigmaSqj = b.second*b.second;

                if((diffSqa/sigmaSqi + diffSqa/sigmaSqj) < 5.99)
                {
                    if(!nearest.count(i))
                    {
                        sumTho += (a.first/sigmaSqi);
                        sumSigmaInv += (1.0/sigmaSqi);
                        nearest.insert(i);
                    }

                    if(!nearest.count(j))
                    {
                        sumTho += (b.first/sigmaSqj);
                        sumSigmaInv += (1.0/sigmaSqj);
                        nearest.insert(j);
                    }
                }
            }
        }

        fusedCount = (int)nearest.size();

        // fuse
        tho = sumTho / sumSigmaInv;
        sigma = sqrt(1.0 / sumSigmaInv);
    }
    return fusedCount;
}


void MapReconstructor::interKeyFrameChecking(KeyFrame* pKF)
{
    cout<<"interKeyFrameChecking" <<endl;
    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints1 = keyframeKeyPointsMap.at(pKF->mnId);
    vector<KeyFrame*> vpNeighKFs = pKF->GetBestCovisibilityKeyFrames(kN);

    cv::Mat R1w = pKF->GetRotation();
    cv::Mat t1w = pKF->GetTranslation();

    vector<RcKeyPoint> validPoints;

    // project neighbour error factors, list of djn, rjiz*xp, tjiz, sigmajnSquare
    map<Point2f,vector<vector<float> > ,Point2fLess> depthErrorEstimateFactors;
    map<Point2f,set<KeyFrame*>,Point2fLess> depthErrorKeyframes;

    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        KeyFrame* pKF2 = vpNeighKFs[i];

        long mnid2 = pKF2->mnId;
        if(!keyframeKeyPointsMap.count(mnid2)){
            cout << "keyframe hypo data not exist: " << mnid2 << endl;
            continue;
        }

        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        cv::Mat R21 = R2w*R1w.t();
        cv::Mat t21 = -R2w*R1w.t()*t1w+t2w;

        cv::Mat KR = pKF2->mK * R21;
        cv::Mat Kt = pKF2->mK * t21;
        cv::Mat rjiz = (cv::Mat_<float>(1,3) << R21.at<float>(2,0), R21.at<float>(2,1), R21.at<float>(2,2));
        float tjiz = t21.at<float>(2);

        map<Point2f,RcKeyPoint,Point2fLess> &keyPoints2 = keyframeKeyPointsMap.at(mnid2);

        for(auto &kpit : keyPoints1)
        {
            RcKeyPoint &kp1 = kpit.second;
            vector<vector<float> > &depthErrorEstimateFactor = depthErrorEstimateFactors[kp1.pt];
            set<KeyFrame*> &depthErrorKeyframe = depthErrorKeyframes[kp1.pt];

            if(!kp1.fused /*|| !kp1.intraCheckCount*/)
            {
                continue;
            }

            float tho1 = kp1.tho;

            // Xba(p) = K.inv() * xp
            cv::Mat xp1 = (cv::Mat_<float>(1,3) << (kp1.pt.x-pKF->cx)*pKF->invfx, (kp1.pt.y-pKF->cy)*pKF->invfy, 1.0);
//            cout<<"xp1 "<<xp1<<endl;
            float invTho = 1.0/tho1;
            cv::Mat D = (Mat_<float>(3,3) << invTho,0,0,0, invTho,0,0,0, invTho);
//            cout<<"D "<<D<<endl;
            Mat KRD = KR*D;
//            cout<<"KRD "<<KRD<<endl;

            // project to neighbour keyframe
            Mat xj = KRD*xp1.t() + Kt;
            float tho2 = tho1 / (rjiz.dot(xp1) + tho1 * tjiz);
//            cout<<"xj "<<xj<<endl;
//            cout<<"tho2 "<<tho2<<endl;

            // check neighbours
            float uj = xj.at<float>(0) / xj.at<float>(2), vj = xj.at<float>(1) / xj.at<float>(2);
            vector<Point2f> neighbours;
            neighbours.push_back(Point2f(floor(uj), floor(vj)));
            neighbours.push_back(Point2f(floor(uj), ceil(vj)));
            neighbours.push_back(Point2f(ceil(uj), floor(vj)));
            neighbours.push_back(Point2f(ceil(uj), ceil(vj)));
            for (const Point2f &p: neighbours)
            {
                if(keyPoints2.count(p))
                {
                    RcKeyPoint &kp2 = keyPoints2.at(p);
                    if(kp2.fused)
                    {
                        float tho2n = kp2.tho;
                        float sigma2nSquare = kp2.sigma * kp2.sigma;

                        // Ka test
                        if((tho2 - tho2n) * (tho2 - tho2n) / sigma2nSquare < 3.84)
                        {
                            kp1.interCheckCount++;
                            validPoints.push_back(kp1);

                            // project neighbour error factors, list of djn, rjiz*xp, tjiz, sigmajnSquar
                            vector<float> params;
                            params.push_back(1.0f/tho2n);
                            params.push_back(rjiz.dot(xp1));
                            params.push_back(tjiz);
                            params.push_back(sigma2nSquare);

                            depthErrorEstimateFactor.push_back(params);
                            depthErrorKeyframe.insert(pKF2);
                        }
                    }
                }
            }
        }
    }

    int validIndex = 0;
    auto estimate = [](vector<vector<float>> &params)
    {
        float sumu = 0.0f, suml = 0.0f;
        for(vector<float> &param: params)
        {
            float djnSquare4 = param[0] * param[0] * param[0] * param[0];
            sumu += param[1] * (param[0] - param[2]) / (djnSquare4 * param[3]);
            suml += param[1] *  param[1]/ (djnSquare4 * param[3]);
        }
        return sumu/suml;
    };
//    bool b = glambda(3, 3.14); // ok
    for(RcKeyPoint &vkp: validPoints)
    {
//        cout<<"vkp.interCheckCount "<<vkp.interCheckCount<<endl;
        size_t count = depthErrorKeyframes.at(vkp.pt).size();
        if(count > lambdaN)
        {
            vector<vector<float>> &params = depthErrorEstimateFactors.at(vkp.pt);
            float dpstar = estimate(params);
                    cout<<"create estimate "<<dpstar<<" former "<<1.0/vkp.tho<<endl;
                    vkp.tho = 1.0f/dpstar;
            addKeyPointToMap(vkp, pKF);
        }
        validIndex++;
    }
}


void MapReconstructor::addKeyPointToMap(RcKeyPoint &kp1, KeyFrame* pKF)
{
    if(kp1.fused){

        const float zh = 1.0/kp1.tho;
//        float error = fabs(kp1.mDepth-zh);
//        if(zh>0) {
//            if(error <= 0.03 * kp1.mDepth * kp1.mDepth /*&& error<0.1 * kp1.mDepth*/){
////                kp1.mDepth = zh;
//                return;
//            }
//            kp1.mDepth = zh;
//        }

        kp1.mDepth = zh;
        if(zh > 8) return;

        cv::Mat x3D = UnprojectStereo(kp1, pKF);
        if(!x3D.empty()){

            MapPoint* pMP = new MapPoint(x3D,pKF,mpMap);
            pMP->UpdateNormalAndDepth();
            mpMap->AddMapPoint(pMP);

        }
    }

    //            pMP->AddObservation(mpCurrentKeyFrame,idx1);
    //            pMP->AddObservation(pKF2,idx2);

    //            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
    //            pKF2->AddMapPoint(pMP,idx2);

    //            pMP->ComputeDistinctiveDescriptors();

    //            pMP->UpdateNormalAndDepth();

    //        mpMap->AddMapPoint(pMP);
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


