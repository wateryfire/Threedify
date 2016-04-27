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

    // camera params
    width = fSettings["Camera.width"];
    height = fSettings["Camera.height"];
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

            // 2. estimate octave
            // judge which interval of one point on edge belongs to, give an approximate estimation of octave
            int octave;
if(!pKF->IsInImage(col,row))continue;
            float depth = depths.at<float>(Point(col, row));
            if(depth<=0 || depth>8)
            {
                continue;
            }
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
//            hgkp.fetchNeighbours(image, gradient);
            hgkp.eachNeighbourCords([&](cv::Point2f ptn){
                vector<float> msg;
                msg.push_back(image.at<float>(ptn));
                msg.push_back(gradient.at<float>(ptn));
                hgkp.neighbours.push_back(msg);
            });
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
            CreateNewMapPoints(currentKeyFrame);

            fuseHypo(currentKeyFrame);

            denoise(currentKeyFrame);

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
//    const vector<KeyFrame*> vpNeighKFs = currentKeyFrame->GetBestCovisibilityKeyFrames(kN);
//    return (int)vpNeighKFs.size() >= kN;
    return mlpKFQueueForReonstruction.size() > 10;
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
    cout<<"median dep"<< tho0<<" "<<sigma0<<" mro "<<medianRotation<<endl;

    // search for each point in first image
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
//        const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
        float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
        const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

        if(a==0&&b==0)
        {
            continue;
        }

        // Xba(p) = K.inv() * xp
        cv::Mat xp1 = (cv::Mat_<float>(1,3) << (kp1.pt.x-pKF1->cx)*pKF1->invfx, (kp1.pt.y-pKF1->cy)*pKF1->invfy, 1.0);
//        cv::Mat xp = (cv::Mat_<float>(3,1) << kp1.pt.x, kp1.pt.y, 1.0);
//        cv::Mat xp1 = pKF1->mK.inv() * xp;

        // upper/lower cordinate bounds in second image
//        float data1[] = {1/thoMax,0,0,0,1/thoMax,0,0,0,1/thoMax};
//        float data2[] = {1/thoMin,0,0,0,1/thoMin,0,0,0,1/thoMin};
//        cv::Mat thoMatMax(3, 3, CV_32FC1, data1);
//        cv::Mat thoMatMin(3, 3, CV_32FC1, data2);
//        cv::Mat thoMatMax = (cv::Mat_<float>(3,3) << 1/thoMax,0,0,0,1/thoMax,0,0,0,1/thoMax);
//        cv::Mat thoMatMin = (cv::Mat_<float>(3,3) << 1/thoMax,0,0,0,1/thoMax,0,0,0,1/thoMax);
//        Mat imgSense0 = pKF2->mK * R21 * pKF2->mK.inv() * thoMatMax + pKF2->mK * t21;
//        float u0 = imgSense0.at<float>(0);
//        Mat imgSense1 = pKF2->mK * R21 * pKF2->mK.inv() * thoMatMin + pKF2->mK * t21;
//        float u1 = imgSense1.at<float>(0);

////////////////////////
        tho0 = 1/kp1.mDepth;
        sigma0 /= 2;
width=pKF2->mnMaxX - pKF2->mnMinX;
height=pKF2->mnMaxY - pKF2->mnMinY;
////////////////////////

        float thoMax = tho0 + 2*sigma0, thoMin = tho0 - 2*sigma0;
        float u0 = pKF1->cx + (rjix.dot(xp1) + thoMax * tjix) / (rjiz.dot(xp1) + thoMax*tjiz) * pKF1->fx;
        float u1 = pKF1->cx + (rjix.dot(xp1) + thoMin * tjix) / (rjiz.dot(xp1) + thoMin*tjiz) * pKF1->fx;

        ////////////////////////
        /// \brief minU
        ///
        //int lowerBoundXInKF2, lowerBoundYInKF2, upperBoundXInKF2, upperBoundYInKF2;
        //bool valid = getSearchAreaForWorld3DPointInKF( pKF1, pKF2, kp1,lowerBoundXInKF2, lowerBoundYInKF2, upperBoundXInKF2, upperBoundYInKF2 );
        //if(!valid) continue;
        //u0=lowerBoundXInKF2;
        //u1=upperBoundXInKF2;
        ///
        /// ////////////////////

        float minU = min(u0, u1), maxU = max(u0, u1);
        float minV = 0, maxV = 0;
//        cout<<"proj bound of u "<<minU<<", "<<maxU<<endl;

        float offset = 1.0, dx, dy;
        //////
        ///
        //minV = lowerBoundYInKF2;
        //maxV = upperBoundYInKF2;
        //offset = 0;
        //b=0;
        ///
        //////
        float offsetU = sqrt(offset * offset * a * a / (a*a + b*b));
        float offsetV = sqrt(offset * offset * b * b / (a*a + b*b));

        bool parralexWithYAxis = (b==0 || fabs(-a / b) > (float)(height/(2*offset)));

        Point2f startCord;
        Point2f endCord;
        if(parralexWithYAxis)
        {
            minV = 0;
            maxV = height;
            //////
            ///
            //minV = lowerBoundYInKF2;
            //maxV = upperBoundYInKF2;
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
                errorSquare = 2;
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
       vector<float> xBound1 = vector<float>(0.98*x3D1, 1.02*x3D1); //todo: to configurable, assume the deviation in (R,t) estimation has 2% portion of accuracy deviation.
       vector<float> yBound1 = vector<float>(0.98*y3D1, 1.02*y3D1); //todo: to configurable, assume the deviation in (R,t) estimation has 2% portion of accuracy deviation.
       for(auto & x: xBound1)
           for(auto & y: yBound1)
               {
                    boundPoints.push_back((cv::Mat_<float>(3,1) << x, y, lowerZ));
                }
       
       upper3Dw = pKF1->UnprojectStereo(u,v, upperZ);
        
        float x3D2=upper3Dw.at<float>(0);
        float y3D2=upper3Dw.at<float>(1);
        vector<float> xBound2 = vector<float>(0.98*x3D2, 1.02*x3D2); //todo: to configurable, assume the deviation in (R,t) estimation has 2% portion of accuracy deviation.
        vector<float> yBound2 = vector<float>(0.98*y3D2, 1.02*y3D2); //todo: to configurable, assume the deviation in (R,t) estimation has 2% portion of accuracy deviation.
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

//    while(angleDiff <0 || angleDiff > 90)
//    {
//        if(angleDiff < 0){
//            angleDiff += 360.0;
//        }else if(angleDiff > 360){
//            angleDiff -= 360.0;
//        }
//        else{
//            if(angleDiff>180){
//                angleDiff-=180.0;
//            }else if(angleDiff>90){
//                angleDiff=180.0 - angleDiff;
//            }
//        }
//    }

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
        if(kp1.fused)
        {
//            cout<<"already fs"<<endl;
//            continue;
        }
        //        cout<<kp1.hypotheses.size()<<endl;
        vector<pair<float, float>> &hypos = kp1.hypotheses;

        set<int> nearest;
        int totalCompact = KaTestFuse(hypos, kp1.tho, kp1.sigma, nearest);

        if(totalCompact<=lambdaN)
        {
            continue;
        }

        kp1.fused = true;

        // set hypotheses fuse flag
        map<RcKeyPoint*,int> &rel = kp1.hypothesesRelation;
        for(auto &kpit2 : rel)
        {
            int fxidx = kpit2.second;
            for(const int fsi : nearest)
            {
                if(fsi == fxidx)
                {
                    kpit2.first->fused = true;
                    break;
                }
            }
        }
    }
}

void MapReconstructor::denoise(KeyFrame* pKF)
{
    long kfid = pKF->mnId;
    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints = keyframeKeyPointsMap.at(kfid);
    for(auto &kpit : keyPoints)
    {
        RcKeyPoint &kp1 = kpit.second;
        //check neighbour hypos
        int neighbourHypos = 0;

        vector<pair<float, float>> nbrhypos;

        kp1.eachNeighbourCords([&](Point2f pt){
            if(keyPoints.count(pt))
            {
                RcKeyPoint &kpn = keyPoints.at(pt);
                if(kpn.fused)
                {
                    nbrhypos.push_back(make_pair(kpn.tho, kpn.sigma));
                    neighbourHypos++;
                }
            }
        });

        if(neighbourHypos<2)
        {
            continue;
        }
        // ka square test
        float tho, sigma;
        set<int> nearest;
        int totalCompact = KaTestFuse(nbrhypos, tho, sigma, nearest);
        if(totalCompact < 2 && kp1.fused)
        {
            kp1.fused = false;
        }
        else if(totalCompact>=2 && !kp1.fused)
        {
            kp1.fused = true;
            kp1.tho = tho;
            kp1.sigma = sigma;
//            kp1.addHypo(tho, sigma, 0);
        }
        addKeyPointToMap(kp1, pKF);
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

void MapReconstructor::addKeyPointToMap(RcKeyPoint &kp1, KeyFrame* pKF)
{
    if(kp1.fused){

        const float zh = 1.0/kp1.tho;
        if(zh>0) {
            if(fabs(kp1.mDepth-zh)<0.05 * kp1.mDepth){
                kp1.mDepth = zh;
            }
        }

        cv::Mat x3D = UnprojectStereo(kp1, pKF);
        if(!x3D.empty()){

            MapPoint* pMP = new MapPoint(x3D,pKF,mpMap);
            mpMap->AddMapPoint(pMP);
            pMP->UpdateNormalAndDepth();

        }
    }

    //            pMP->AddObservation(mpCurrentKeyFrame,idx1);
    //            pMP->AddObservation(pKF2,idx2);

    //            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
    //            pKF2->AddMapPoint(pMP,idx2);

    //            pMP->ComputeDistinctiveDescriptors();

    //            pMP->UpdateNormalAndDepth();

    //        mpMap->AddMapPoint(pMP);
    //            mlpRecentAddedMapPoints.push_back(pMP);
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


