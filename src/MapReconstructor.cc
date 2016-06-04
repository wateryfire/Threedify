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
float errorTolerenceFactor,initVarienceFactor,depthGradientThres = 255.0;
bool measuredDepthConstraient = true;
bool needRectify = false;
float baselineThres;

map<size_t, SparseMat_<MapReconstructor::RcHighGradientPoint*> > keyFrameHighGradientPointsMat;

MapReconstructor::MapReconstructor(Map* pMap,  const string &strSettingPath):
        mpMap(pMap)
{
	mStatus_KeyFrameQueueProcess=INITIALIZED;
	mStatus_RealTimeMapReconstruction=INITIALIZED;
	mStatus_FullMapReconstruction=INITIALIZED;

    // Get re-construction params from settings file
    //todo: Exception handling and default value
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    int fsKN = fSettings["ReConstruction.KN"];
    if(fsKN > 0)
    {
        mKN = fsKN;
    }

    int fsLambdaN = fSettings["ReConstruction.lambdaN"];
    if(fsLambdaN > 0)
    {
        mLambdaN = fsLambdaN;
    }

    float fsSigmaI = fSettings["ReConstruction.sigmaI"];
    if(fsKN > 0)
    {
        mSigmaI = fsSigmaI;
    }

    float fsLambdaG = fSettings["ReConstruction.lambdaG"];
    if(fsLambdaG>0)
    {
        mLambdaG = fsLambdaG;
    }

    float fsLambdaL = fSettings["ReConstruction.lambdaL"];
    if(fsLambdaL>0)
    {
        mLambdaL = fsLambdaL;
    }

    float fsLambdaThe = fSettings["ReConstruction.lambdaThe"];
    if(fsLambdaThe>0)
    {
        mLambdaThe = fsLambdaThe;
    }

    float fsTheta = fSettings["ReConstruction.theta"];
    if(fsTheta>0)
    {
        mTheta = fsTheta;
    }

    // camera params
    mWidth = fSettings["Camera.width"];
    mHeight = fSettings["Camera.height"];

    //
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    //DEBUG
    mDepthThresholdMax = fSettings["ReConstruction.maxDepth"];
    mDepthThresholdMin = fSettings["ReConstruction.minDepth"];
    mEpipolarSearchOffset = fSettings["ReConstruction.searchOffset"];
    errorTolerenceFactor = fSettings["ReConstruction.errorTolerenceFactor"];
    initVarienceFactor = fSettings["ReConstruction.initVarienceFactor"];
    measuredDepthConstraient = ((int)fSettings["ReConstruction.disableMeasuredDepthConstraient"] ==0);
    depthGradientThres = fSettings["ReConstruction.depthGradientThres"];
    needRectify = ((int)fSettings["ReConstruction.rectify"]==1);
    baselineThres = fSettings["ReConstruction.baselineThres"];
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
//        currentKeyFrame->mRefImgGray.deallocate();
//        currentKeyFrame->mRefImgDepth.deallocate();

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
    normalize(modulo, modulo, 0x00, 0xFF, NORM_MINMAX, CV_32F);

    HighGradientAreaPoints(modulo,orientation, pKeyFrame, mLambdaG);
}

void MapReconstructor::HighGradientAreaPoints(Mat &gradient, Mat &orientation, KeyFrame *pKF, const float gradientThreshold)
{
    Mat &image = pKF->mRefImgGray;
    Mat &depths = pKF->mRefImgDepth;

//    map<Point2f,RcKeyPoint,Point2fLess> keyPoints;
    int dims = 2;
    int size[] = {mHeight, mWidth};
    SparseMat_<RcHighGradientPoint*> highGradientPoints = SparseMat_<RcHighGradientPoint*>(dims, size);
    map<Point2f, vector<float>*, Point2fLess> neighbourMsgCache;

    set<Point,Point2fLess> depthCurvature = DepthCurvatureFilter(depths);

    int matsize=0;
    for(int row = 0; row < image.rows; ++row)
    {
        uchar* p = image.ptr<uchar>(row);
        float* pd = depths.ptr<float>(row);
        float* pg = gradient.ptr<float>(row);
        float* po = orientation.ptr<float>(row);

        for(int col = 0; col < image.cols; ++col) {
            float intensity = p[col];
            Point cord = Point(col, row);

            if(!pKF->IsInImage(col, row))
            {
                continue;
            }

            float depth = pd[col];
            if(depth<=mDepthThresholdMin || depth > mDepthThresholdMax)
            {
                continue;
            }

            // 1. filter with intensity threshold
            float gradientModulo = pg[col];
            if((gradientModulo < gradientThreshold) && !(depthCurvature.count(cord) && gradientModulo>mTheta))
            {
                continue;
            }

            float angle = po[col];
//            cout<<"intensity "<<saturate_cast<float>(intensity)<<" angle "<<angle<<" gradientModulo "<<saturate_cast<float>(gradientModulo)<<endl;
            RcHighGradientPoint *hgkp = new RcHighGradientPoint(col, row,intensity,gradientModulo,angle,0,depth);

            // fill neighbour info
            bool outOfRange = false;
            hgkp->eachNeighbourCords([&](float x, float y)
            {
                if(CordInImageBounds(x, y, mWidth, mHeight))
                {
                    Point2f pt = Point2f(x, y);
                    if(neighbourMsgCache.count(pt))
                    {
                        vector<float> *msg = neighbourMsgCache.at(pt);
                        hgkp->neighbours.push_back(*msg);
                    }
                    else
                    {
                        vector<float> *msg = new vector<float>(2);;
                        uchar nIntensity = image.ptr<uchar>(pt.y)[(int)pt.x];
                        float nGradientModulo = gradient.ptr<float>(pt.y)[(int)pt.x];
                        vector<float> &msgr = *msg;
                        msgr[0] = nIntensity;
                        msgr[1] = nGradientModulo;
                        hgkp->neighbours.push_back(msgr);
                        neighbourMsgCache[pt] = msg;
                    }
                }
                else
                {
                    outOfRange = true;
                }
            });
            if(outOfRange)
            {
                delete hgkp;
                continue;
            }

            // undistort
//            cv::Mat mat(1,2,CV_32F);
//            mat.at<float>(0,0)=hgkp.pt.x;
//            mat.at<float>(0,1)=hgkp.pt.y;

//            mat=mat.reshape(2);
//            cv::undistortPoints(mat,mat,pKF->mK,mDistCoef,cv::Mat(), pKF->mK);
//            mat=mat.reshape(1);

//            hgkp.ptu.x=mat.at<float>(0,0);
//            hgkp.ptu.y=mat.at<float>(0,1);

//            keyPoints[cord] = hgkp;
            highGradientPoints.ref(row, col) = hgkp;
            matsize++;
        }
    }
    cout<<"highGradientPoints size "<<matsize<<endl;
//    keyframeKeyPointsMap[pKF->mnId] = keyPoints;
    keyFrameHighGradientPointsMat[pKF->mnId] = highGradientPoints;
}

set<Point,MapReconstructor::Point2fLess> MapReconstructor::DepthCurvatureFilter(Mat &depths)
{
    set<Point,Point2fLess> cords;

    cv::Mat Sx;
    cv::Scharr( depths, Sx, CV_32F, 1, 0);

    cv::Mat Sy;
    cv::Scharr( depths, Sy, CV_32F, 0, 1);

    int statStep = 14, signTurnThres=2;
    int turnThres = (signTurnThres * statStep * 0.1);

    int width = depths.size().width;
    int height = depths.size().height;

    int offset = 2;

    for ( int v = 0; v < height; v++ )
    {
        int majorDir = 1;
        int possitiveCount = statStep;
        for(int u=0;u<width;u++)
        {
            float depth = depths.at<float>(v, u);
            if(depth<0)
            {
                continue;
            }

            float quadent=Sx.at<float>(v,u);
            if((quadent * majorDir < 0.0) && (possitiveCount > 0))
            {
                possitiveCount--;
            }
            else if((quadent * majorDir >= 0.0) && (possitiveCount < statStep))
            {
                possitiveCount++;
            }
            else
            {
                continue;
            }

            if(possitiveCount<turnThres)
            {
                //inverse
                majorDir *=-1;
                possitiveCount = (statStep - possitiveCount);

                for(int c = u-possitiveCount - offset;c < u-possitiveCount + offset+1;c++)
                {
                    cords.insert(cv::Point( c, v));
                }
            }

//            if(quadent > 0.5)
//            {
//                cords.insert(cv::Point( u, v ));
//            }
        }
    }


    for ( int u = 0; u < width; u++ )
    {
        int majorDir = 1;
        int possitiveCount = statStep;
        for(int v=0; v< height; v++)
        {
            float depth = depths.at<float>(v, u);
            if(depth<0)
            {
                continue;
            }

            float quadent=Sy.at<float>(v, u);
            if((quadent * majorDir < 0.0) && (possitiveCount > 0))
            {
                possitiveCount--;
            }
            else if((quadent * majorDir >= 0.0) && (possitiveCount < statStep))
            {
                possitiveCount++;
            }
            else
            {
                continue;
            }

            if(possitiveCount<turnThres)
            {
                //inverse
                majorDir *=-1;
                possitiveCount = (statStep - possitiveCount);
                for(int c = v-possitiveCount - offset;c < v-possitiveCount + offset+1;c++)
                {
                    cords.insert(cv::Point( u, c));
                }
            }

//            if(quadent > 0.5)
//            {
//                cords.insert(cv::Point( u, v ));
//            }
        }
    }
    return cords;
}

void MapReconstructor::RunToReconstructMap()
{
    mRealTimeReconstructionEnd = false;

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

        bool frameValid = (mlpKFQueueForReonstruction.size() > (size_t)10);
//        if (frameValid || !keyframeKeyPointsMap.count(currentKeyFrame->mnId))
        if (frameValid || !keyFrameHighGradientPointsMat.count(currentKeyFrame->mnId))
        {
            //Get Mutex-lock to access the queue of key frames.
            {
                unique_lock<mutex> lock(mMutexForKFQueueForReonstruction);
                mlpKFQueueForReonstruction.pop_front();
                interKeyFrameCheckingStack.push_back(currentKeyFrame);
            }
            retryCount=0;

//            if(keyframeKeyPointsMap.count(currentKeyFrame->mnId))
            if(keyFrameHighGradientPointsMat.count(currentKeyFrame->mnId))
            {
//                map<Point2f,RcKeyPoint,Point2fLess> &keyPoints1 = keyframeKeyPointsMap.at(currentKeyFrame->mnId);

    //            for(auto &kpit : keyPoints1)
    //            {
    //                RcKeyPoint &kp1 = kpit.second;
    //                kp1.tho = 1.0f / kp1.mDepth;
    //                kp1.sigma = initVarienceFactor/(1.0 - initVarienceFactor) * kp1.mDepth;
    //                kp1.intraCheckCount  = 1;
    //                kp1.fused = true;
    //            }

                CreateNewMapPoints(currentKeyFrame);

                FuseHypo(currentKeyFrame);

                IntraKeyFrameChecking(currentKeyFrame);
            }
        }
        else
        {
            retryCount ++;
            usleep(30000);
            continue;
        }

        while(interKeyFrameCheckingStack.size() > (size_t)mKN)
//        while(interKeyFrameCheckingStack.size() > 0)
        {
            currentKeyFrameInterChecking = interKeyFrameCheckingStack.front();
//            if(keyframeKeyPointsMap.count(currentKeyFrameInterChecking->mnId))
            if(keyFrameHighGradientPointsMat.count(currentKeyFrameInterChecking->mnId))
            {
                InterKeyFrameChecking(currentKeyFrameInterChecking);
            }
            interKeyFrameCheckingStack.pop_front();
        }
    }

    mRealTimeReconstructionEnd = true;

	cout << "MapReconstructor: End thread execution for map reconstruction during SLAM tracking." << endl;

	cout << "MapReconstructor: Start thread execution for full map reconstruction." << endl;

	// TODO: Remove the sleep process, once the real code is implemented.
//	usleep(10000);

	cout << "MapReconstructor: End thread execution for full map reconstruction." << endl;
}

bool MapReconstructor::isRealTimeReconstructionEnd()
{
    return mRealTimeReconstructionEnd;
}

bool MapReconstructor::CheckNewKeyFrames(KeyFrame* currentKeyFrame)
{
//    const vector<KeyFrame*> vpNeighKFs = currentKeyFrame->GetBestCovisibilityKeyFrames(kN);
//    return (int)vpNeighKFs.size() >= kN;
    return mlpKFQueueForReonstruction.size() > (size_t)10;
}

void MapReconstructor::CreateNewMapPoints(KeyFrame* mpCurrentKeyFrame)
{
    cout<<"CreateNewMapPoints"<<endl;
    // Retrieve neighbor keyframes in covisibility graph
    int nn = mKN;
    nn*=2;

    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    // Search matches with epipolar restriction and triangulate
    int count=0;
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        long mnid2 = pKF2->mnId;
//        if(!keyframeKeyPointsMap.count(mnid2))
        if(!keyFrameHighGradientPointsMat.count(mnid2))
        {
            cout << "keyframe data not extracted yet: " << mnid2 << endl;
            continue;
        }

        cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
        const float ratioBaselineDepth = baseline/medianDepthKF2;

        if(ratioBaselineDepth<baselineThres)
        {
            continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);
        EpipolarConstraientSearch(mpCurrentKeyFrame, pKF2, F12,vMatchedIndices);

        count++;
        if(count==mKN)
        {
            break;
        }
    }
}

cv::Mat MapReconstructor::UnprojectStereo(RcHighGradientPoint &p,KeyFrame *pKF)
{
    float z = p.mDepth;
    const float u = p.pt.x;
    const float v = p.pt.y;
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

void MapReconstructor::EpipolarConstraientSearch(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,vector<pair<size_t,size_t> > &vMatchedIndices)
{
    // get rotation (j - i)
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R21 = R2w*R1w.t();
    cv::Mat t21 = -R2w*R1w.t()*t1w+t2w;

    // R-t params on each axis
    cv::Mat rjiz = (cv::Mat_<float>(1,3) << R21.at<float>(2,0), R21.at<float>(2,1), R21.at<float>(2,2));
    cv::Mat rjix = (cv::Mat_<float>(1,3) << R21.at<float>(0,0), R21.at<float>(0,1), R21.at<float>(0,2));
    cv::Mat rjiy = (cv::Mat_<float>(1,3) << R21.at<float>(1,0), R21.at<float>(1,1), R21.at<float>(1,2));
    float tjiz = t21.at<float>(2);
    float tjix = t21.at<float>(0);
    float tjiy = t21.at<float>(1);

    // high gradient area points
    const long mnid1 = pKF1->mnId, mnid2 = pKF2->mnId;
    cout<<"epipcolar constraient search between "<<mnid1<<" "<<mnid2<<endl;
//    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints1 = keyframeKeyPointsMap.at(mnid1);
//    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints2 = keyframeKeyPointsMap.at(mnid2);
    SparseMat_<RcHighGradientPoint*> &highGradientPoints1 = keyFrameHighGradientPointsMat.at(mnid1);
    SparseMat_<RcHighGradientPoint*> &highGradientPoints2 = keyFrameHighGradientPointsMat.at(mnid2);

    // get median rotation between KFs
    float medianRotation = CalcMedianRotation(pKF1,pKF2);

    // inverse depth of sense
    float tho0, sigma0;
    CalcSenceInverseDepthBounds(pKF1, tho0, sigma0);
    cout<<"median dep tho(1/d): "<< tho0<<" sigma: "<<sigma0<<" median rotation "<<medianRotation<<endl;

    // search for each point in first image
//    for(auto &kpit : keyPoints1)
    SparseMatIterator_<RcHighGradientPoint*>
        it = highGradientPoints1.begin(),
        it_end = highGradientPoints1.end();
    for(; it != it_end; ++it)
    {
//        RcKeyPoint &kp1 = *(it.ref<RcKeyPoint*>());
//        RcKeyPoint &kp1 = kpit.second;
        RcHighGradientPoint &kp1 = **it;

//        if(kp1.fused)
//        {
//            highGradientPoints1.ref(kp1.pt.y, kp1.pt.x) = NULL;
//            delete &kp1;
//            continue;
//        }

        // prepare data
        float intensity1 = kp1.intensity;
        float gradient1 = kp1.gradient;

        Point2f matchedCord;

        float xudist, yudist;
        if(needRectify)
        {
            // undistort
            cv::Mat mat(1,2,CV_32F);
            mat.at<float>(0,0)=kp1.pt.x;
            mat.at<float>(0,1)=kp1.pt.y;

            mat=mat.reshape(2);
            cv::undistortPoints(mat,mat,pKF1->mK,mDistCoef,cv::Mat(), pKF1->mK);
            mat=mat.reshape(1);

            xudist=mat.at<float>(0,0);
            yudist=mat.at<float>(0,1);
        }
        else
        {
            xudist = kp1.pt.x;
            yudist = kp1.pt.y;
        }

        // epipolar line params
        const float a = xudist*F12.at<float>(0,0)+yudist*F12.at<float>(1,0)+F12.at<float>(2,0);
        const float b = xudist*F12.at<float>(0,1)+yudist*F12.at<float>(1,1)+F12.at<float>(2,1);
        const float c = xudist*F12.at<float>(0,2)+yudist*F12.at<float>(1,2)+F12.at<float>(2,2);

        if(a==0&&b==0)
        {
            continue;
        }

        float sigmaEst = sigma0;

        // relocalize search area under depth estimate
        ////////////////////////
        if(measuredDepthConstraient)
        {
            tho0 = 1.0/kp1.mDepth;
            sigmaEst = min((1.0f/(1.0f - initVarienceFactor) / kp1.mDepth - tho0) / 2.0f, sigma0);
        }
        ////////////////////////

        // inverse depth bounds
        float thoMax = tho0 + 2.0f*sigmaEst, thoMin = tho0 - 2.0f*sigmaEst;

        // unary ray through xp : Xba(p) = K.inv() * xp
        cv::Mat xp1 = (cv::Mat_<float>(1,3) << (xudist-pKF1->cx)*pKF1->invfx, (yudist-pKF1->cy)*pKF1->invfy, 1.0);
        // bounds of x axis
        float u0 = pKF1->cx + (rjix.dot(xp1) + thoMax * tjix) / (rjiz.dot(xp1) + thoMax*tjiz) * pKF1->fx;
        float u1 = pKF1->cx + (rjix.dot(xp1) + thoMin * tjix) / (rjiz.dot(xp1) + thoMin*tjiz) * pKF1->fx;
        // bounds of y axis
        float v0 = pKF1->cy + (rjiy.dot(xp1) + thoMax * tjiy) / (rjiz.dot(xp1) + thoMax*tjiz) * pKF1->fy;
        float v1 = pKF1->cy + (rjiy.dot(xp1) + thoMin * tjiy) / (rjiz.dot(xp1) + thoMin*tjiz) * pKF1->fy;

        float minSimilarityError = MatchAlongEpipolarLine(matchedCord, kp1, highGradientPoints2, medianRotation, u0, u1, v0, v1, a, b, c);

        // use the best match point to estimate the distribution
        if(minSimilarityError >=0 && minSimilarityError<1.0e+3)
        {
//            RcKeyPoint &match = keyPoints2.at(matchedCord);
            RcHighGradientPoint &match = *highGradientPoints2.ref(matchedCord.y, matchedCord.x);

            // subpixel estimation:
            // approximate intensity gradient along epipolar line : g=(I(uj + 1)-I(uj - 1))/2;
            // approximate intensity gradient module derivate along epipolar line : q=(G(uj + 1)-G(uj - 1))/2;
            // sub pixel estimation:
            // u0star = u0 + (g(u0)*ri(u0) + 1/theta * q(u0) * rg(u0)) / (g(u0) * g(u0) + 1/theta * q(u0) * q(u0) )
            // sigmau0star^2 = 2* si*si / (g(u0) * g(u0) + 1/theta * q(u0) * q(u0))
            float u0 = match.pt.x;

            // neighbour info of this point
            vector<float> upper,lower;
            match.getNeighbourAcrossLine(a,b,lower,upper);

            float intensityUpper = upper[0], gradientUpper = upper[1];
            float intensityLower = lower[0], gradientLower = lower[1];

            // derivate along epipolar line
            float g = (intensityUpper - intensityLower) / 2.0;
            float q = (gradientUpper - gradientLower) / 2.0;

            // intensity/gradient error
            float intensityError = intensity1 - match.intensity;
            float gradientError = gradient1 - match.gradient;

            // subpixel estimation of u0
            float errorSquare = (g*g + 1.0/mTheta *q*q);
            float leftPart = (g * intensityError + 1.0 / mTheta * q * gradientError ) / errorSquare;
            if(fabs(leftPart)>1)
            {
                leftPart = 0;
                errorSquare = max(initVarienceFactor, errorSquare);
            }
            float u0Star = u0 + leftPart;
            float sigmaU0Star = sqrt( 2.0 * mSigmaI * mSigmaI /errorSquare );

            // inverse depth hypothese
            float rho = (rjiz.dot(xp1) *(u0Star-pKF1->cx) - pKF1->fx * rjix.dot(xp1) ) / (-tjiz * (u0Star-pKF1->cx) + pKF1->fx * tjix );
            if(isnan(rho))
            {
                continue;
            }

            float u0Starl = u0Star - sigmaU0Star, u0Starr = u0Star + sigmaU0Star;
            float rhoUpper = (rjiz.dot(xp1) *(u0Starr-pKF1->cx) - pKF1->fx * rjix.dot(xp1) ) / (-tjiz * (u0Starr-pKF1->cx) + pKF1->fx * tjix );
            float rhoLower = (rjiz.dot(xp1) *(u0Starl-pKF1->cx) - pKF1->fx * rjix.dot(xp1) ) / (-tjiz * (u0Starl-pKF1->cx) + pKF1->fx * tjix );
            float sigmaRho = max(fabs(rhoUpper - rho),fabs(rhoLower - rho));

//            cout<<"addHypo " <<rho << " " << sigmaRho<<endl;
            kp1.addHypo(rho, sigmaRho,&match);
        }
    }
}

//float MapReconstructor::MatchAlongEpipolarLine(Point2f &matchedCord, RcKeyPoint &kp1, map<Point2f,RcKeyPoint,Point2fLess> &keyPoints2, float &medianRotation, float &u0 ,float &u1, float &v0, float &v1, const float &a, const float &b, const float &c)
float MapReconstructor::MatchAlongEpipolarLine(Point2f &matchedCord, RcHighGradientPoint &kp1, SparseMat_<RcHighGradientPoint*> &highGradientPoints2, float &medianRotation, float &u0 ,float &u1, float &v0, float &v1, const float &a, const float &b, const float &c)
{
    float minSimilarityError = -1;
    float minU = min(u0, u1), maxU = max(u0, u1);
    float minV = 0, maxV = 0;

    float offset = mEpipolarSearchOffset, dx, dy;
    float offsetU = sqrt(2.0f * offset * offset * a * a / (a*a + b*b));
    float offsetV = sqrt(2.0f * offset * offset * b * b / (a*a + b*b));

    bool paralledWithYAxis = (b==0 || fabs(-a / b) > (float)(mHeight/(2*offset)));

    Point2f startCord;
    Point2f endCord;
    if(paralledWithYAxis)
    {

        minV = min(v0,v1);
        maxV = max(v0,v1);
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

    if(!CordInImageBounds(startCord.x,startCord.y,mWidth,mHeight))
    {
        bool bounds = CalCordBounds(startCord, endCord, mWidth, mHeight, a, b, c);
        if(!bounds)
        {
            return minSimilarityError;
        }
        else
        {
            minU = max(startCord.x, minU);
            maxU = min(maxU, endCord.x);
            if(!paralledWithYAxis)
            {
                minV = -(c + a * minU) / b;
                maxV = -(c + a * minU) / b;
            }
            else
            {
                minV = max(min(startCord.y,endCord.y), minV);
                maxV = min(max(startCord.y,endCord.y), minV);
            }
        }
    }

    minU -= offsetU;
    maxU += offsetU;
    minV -= offsetV;
    maxV += offsetV;
    startCord.x = minU;
    startCord.y = minV;

    Point2f cordP;
    while(startCord.x < (maxU + 1.0))
    {
        float x = startCord.x, y = startCord.y;
        while(y<(maxV + 1.0))
        {
            Point2f disp = Point2f(x,y);
//            if(needRectify)
//            {
//                Distort(disp, pKF2);
//            }
            cordP.x = round(disp.x);
            cordP.y= round(disp.y);
//            if(keyPoints2.count(cordP))
            if(highGradientPoints2.ptr(cordP.y, cordP.x, false)  != NULL)
            {
//                RcKeyPoint &kp2 = keyPoints2.at(cordP);
                RcHighGradientPoint &kp2 = *highGradientPoints2.ref(cordP.y, cordP.x);
                //if(!kp2.fused)
                //{
                float similarityError = CheckEpipolarLineConstraient(kp1, kp2, a, b, c ,medianRotation);

                // update the best match point
                if((minSimilarityError < 0 && similarityError >=0) || minSimilarityError > similarityError)
                {
                    minSimilarityError = similarityError;
                    matchedCord.x = cordP.x;
                    matchedCord.y = cordP.y;
                }
                //}
            }

            y += 1.0;
        }

        startCord.x += dx;
        startCord.y += dy;
        if(!paralledWithYAxis)
        {
            maxV += dy;
        }
    }
    return minSimilarityError;
}

void MapReconstructor::Distort(Point2f &point, KeyFrame* pKF)
{
    // To relative coordinates
    float x = (point.x - pKF->cx) / pKF->fx;
    float y = (point.y - pKF->cy) / pKF->fy;

    float r2 = x*x + y*y;
    float k1= mDistCoef.at<float>(0), k2= mDistCoef.at<float>(1),p1= mDistCoef.at<float>(2),p2= mDistCoef.at<float>(3);
    float k3 = mDistCoef.rows > 4 ? mDistCoef.at<float>(4) : 0;
    // Radial distorsion
    float xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    // Back to absolute coordinates.
    xDistort = xDistort * pKF->fx + pKF->cx;
    yDistort = yDistort * pKF->fy + pKF->cy;

    point.x = xDistort;
    point.y = yDistort;
}

bool MapReconstructor::GetSearchAreaForWorld3DPointInKF ( KeyFrame* const  pKF1, KeyFrame* const pKF2, const RcHighGradientPoint& twoDPoint,float& u0, float& v0, float& u1, float& v1, float& offsetU, float& offsetV )
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

        float ZcBound[] = {(float)0.95*z, (float)1.05*z};  //todo: to configurable, assume the depth estimation has 5% portion of accuracy deviation
   
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
        valid = pKF2->ProjectStereo(P3Dw1, u1, v1);
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

bool MapReconstructor::CalCordBounds(Point2f &startCordRef, Point2f &endCordRef, float width, float height, float a, float b,float c)
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
        if(!isnan(xc) && !isnan(yc) && CordInImageBounds(xc,yc,width,height))
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

float MapReconstructor::CheckEpipolarLineConstraient(RcHighGradientPoint &kp1, RcHighGradientPoint &kp2, float a, float b, float c, float medianRotation)
{
    float similarityError = -1.0;
    // prepare data
    float intensity1 = kp1.intensity;
    float angle1 = kp1.orientation;
    float gradient1 = kp1.gradient;

    float intensity2 = kp2.intensity;
    float angle2 = kp2.orientation;
    float gradient2 = kp2.gradient;

    // check epipolar line angle with orientation
    float eplAngle = fastAtan2(-a,b);
    float eplAngleDiff  = angle2 - eplAngle ;
    while(eplAngleDiff <0.0)
    {
        eplAngleDiff += 360.0;
    }
    while(eplAngleDiff >360.0)
    {
        eplAngleDiff -= 360.0;
    }
    if((mLambdaL <= eplAngleDiff && eplAngleDiff < (180.0-mLambdaL)) || ((180.0 + mLambdaL) <= eplAngleDiff && eplAngleDiff < (360.0-mLambdaL)) )
    {
        return similarityError;
    }

    // check in-plane rotation
    float angleDiff = angle2 - angle1 - medianRotation;
    while(angleDiff <0.0)
    {
        angleDiff += 360.0;
    }
    while(angleDiff >360.0)
    {
        angleDiff -= 360.0;
    }

//    angleDiff = fabs(angleDiff);

//    if(angleDiff >= lambdaThe)
//    {
//        return similarityError;
//    }

    if((mLambdaThe <= angleDiff && angleDiff < (180.0-mLambdaThe)) || ((180.0 + mLambdaThe) <= angleDiff && angleDiff < (360.0-mLambdaThe)) )
    {
        return similarityError;
    }

    // cal similarity error
    float intensityError = intensity1 - intensity2;
    float gradientError = gradient1 - gradient2;
    similarityError = (intensityError * intensityError + 1.0/mTheta * gradientError * gradientError ) / ( mSigmaI * mSigmaI );
    return similarityError;
}

float MapReconstructor::CalcMedianRotation(KeyFrame* pKF1,KeyFrame* pKF2)
{
//    const map<Point2f,RcKeyPoint,Point2fLess> &keyPoints1 = keyframeKeyPointsMap.at(pKF1->mnId);
//    const map<Point2f,RcKeyPoint,Point2fLess> &keyPoints2 = keyframeKeyPointsMap.at(pKF2->mnId);

    const std::vector<cv::KeyPoint> &mvKeys1 = pKF1->mvKeys;
    const std::vector<cv::KeyPoint> &mvKeys2 = pKF2->mvKeys;

    vector<float> angles;
    float median = 0.0;

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
            float angleDiff = kp2.angle - kp1.angle;
//            while(angleDiff <0.0)
//            {
//                angleDiff += 360.0;
//            }
//            while(angleDiff >360.0)
//            {
//                angleDiff -= 360.0;
//            }
            angles.push_back(angleDiff);
//            Point2f c1 = Point2f(kp1.pt.x, kp1.pt.y);
//            Point2f c2 = Point2f(kp2.pt.x, kp2.pt.y);
//            if(keyPoints1.count(c1) && keyPoints2.count(c2))
//            {
//                float ort2 = keyPoints2.at(c2).orientation, ort1 = keyPoints1.at(c1).orientation;
//                angles.push_back(ort2 - ort1);
//            }
        }
    }

    if(angles.empty())
    {
        return median;
    }

    size_t size = angles.size();

    sort(angles.begin(), angles.end());

    // quadrant denoise
    int quadrant = 6;
    float halfQuadrantSize = 180.0 / quadrant;
    size_t upper = size - 1, lower = 0;
    float diff = 180.0;

    while(diff > halfQuadrantSize)
    {
        if (size  % 2 == 0)
        {
            median = (angles[(upper + lower + 1) / 2 - 1] + angles[(upper + lower + 1) / 2]) / 2;
        }
        else
        {
            median = angles[(upper + lower + 1) / 2];
        }

        float diffU = angles[upper] - median, diffL = angles[lower] - median;
        if(fabs(diffU) > fabs(diffL))
        {
            upper --;
            diff = diffU;
        }
        else
        {
            lower --;
            diff = diffL;
        }
        size --;

        if(size < (size_t)quadrant)
        {
            break;
        }
    }

    if(median>360.0)
    {
        cout<<"invalid median rotation"<<endl;
        for(float diff:angles)
        {
            cout<<"mp angle diff "<<diff<<endl;
        }
        median = 0.0;
    }

    return median;
}

void MapReconstructor::CalcSenceInverseDepthBounds(KeyFrame* pKF, float &tho0, float &sigma0)
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

bool MapReconstructor::CordInImageBounds(float x, float y, int width, int height)
{
    return (x>=0.0 && x<=width && y>=0.0 && y<=height);
}

void MapReconstructor::FuseHypo(KeyFrame* pKF)
{
    cout<< "enter fuse"<<endl;
    long kfid = pKF->mnId;
//    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints = keyframeKeyPointsMap.at(kfid);
    SparseMat_<RcHighGradientPoint*> &highGradientPoints = keyFrameHighGradientPointsMat.at(kfid);
//    for(auto &kpit : keyPoints)
    SparseMatIterator_<RcHighGradientPoint*>
        it = highGradientPoints.begin(),
        it_end = highGradientPoints.end();
    for(; it != it_end; ++it)
    {
//        RcKeyPoint &kp1 = kpit.second;
//        RcKeyPoint &kp1 = *(it.value<RcKeyPoint*>());
        RcHighGradientPoint &kp1 = **it;
        if(!kp1.hasHypo) continue;
//        cout<<(int)kp1.intensity<<endl;
        vector<pair<float, float>> &hypos = kp1.hypotheses;
//        cout<<(int)hypos.size()<<" "<<hypos.empty()<<endl;

        // DEBUG
//        hypos.push_back(make_pair(1/kp1.mDepth, kp1.mDepth * 0.01));

        set<int> nearest;
        int totalCompact = KaTestFuse(hypos, kp1.tho, kp1.sigma, nearest);

        if(totalCompact<=mLambdaN)
        {
            continue;
        }

        kp1.fused = true;

        // set hypotheses fuse flag
//        map<RcHighGradientPoint*,int> &rel = kp1.hypothesesRelation;
//        for(auto &kpit2 : rel)
//        {
//            int fxidx = kpit2.second;
//            for(const int fsi : nearest)
//            {
//                if(fsi == fxidx)
//                {
//                    RcHighGradientPoint* pkp2 = kpit2.first;
//                    pkp2->fused = true;
////                    delete pkp2;
//                }
//            }
//        }
    }
}

void MapReconstructor::IntraKeyFrameChecking(KeyFrame* pKF)
{
    cout<<"intraKeyFrameChecking "<<endl;
    long kfid = pKF->mnId;
//    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints = keyframeKeyPointsMap.at(kfid);
    SparseMat_<RcHighGradientPoint*> &highGradientPoints = keyFrameHighGradientPointsMat.at(kfid);
    //for(auto &kpit : keyPoints)
    SparseMatIterator_<RcHighGradientPoint*>
        it = highGradientPoints.begin(),
        it_end = highGradientPoints.end();
    for(; it != it_end; ++it)
    {
//        RcKeyPoint &kp1 = kpit.second;
//        RcKeyPoint &kp1 = *(it.value<RcKeyPoint*>());
        RcHighGradientPoint &kp1 = **it;
//        RcKeyPoint &kp1 = kpit.second;
        //check neighbour hypos
        int neighbourHypos = 0;

        vector<pair<float, float>> nbrhypos;

        vector<int> matchedIndexes;
        int index = 0;
        kp1.eachNeighbourCords([&](float x, float y)
        {
//            if(keyPoints.count(pt))
//            {
//                RcKeyPoint &kpn = keyPoints.at(pt);
            if(highGradientPoints.ptr(y, x, false) !=NULL)
            {
                RcHighGradientPoint &kpn = *highGradientPoints.ref(y, x);
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
                int qDiff  =abs(matchedIndexes.back() - matchedIndexes.front());
                if(qDiff < 3 || qDiff > 5)
                {
                    continue;
                }
                kp1.fused = true;
                kp1.tho = tho;
                kp1.sigma = sigma;
            }

            kp1.intraCheckCount = totalCompact;
//            addKeyPointToMap(kp1, pKF);
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


void MapReconstructor::InterKeyFrameChecking(KeyFrame* pKF)
{
    cout<<"interKeyFrameChecking" <<endl;
//    map<Point2f,RcKeyPoint,Point2fLess> &keyPoints1 = keyframeKeyPointsMap.at(pKF->mnId);
    SparseMat_<RcHighGradientPoint*> &highGradientPoints1 = keyFrameHighGradientPointsMat.at(pKF->mnId);
    int nn = mKN;
    nn*=2;
    vector<KeyFrame*> vpNeighKFs = pKF->GetBestCovisibilityKeyFrames(nn);

    cv::Mat R1w = pKF->GetRotation();
    cv::Mat t1w = pKF->GetTranslation();

    vector<RcHighGradientPoint> validPoints;

    // project neighbour error factors, list of djn, rjiz*xp, tjiz, sigmajnSquare
    map<Point2f,vector<vector<float> > ,Point2fLess> depthErrorEstimateFactors;
    map<Point2f,set<KeyFrame*>,Point2fLess> depthErrorKeyframes;

    int count = 0;
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        KeyFrame* pKF2 = vpNeighKFs[i];

        long mnid2 = pKF2->mnId;
//        if(!keyframeKeyPointsMap.count(mnid2))
        if(!keyFrameHighGradientPointsMat.count(mnid2))
        {
            cout << "keyframe hypo data not exist: " << mnid2 << endl;
            continue;
        }

        cv::Mat Ow1 = pKF->GetCameraCenter();
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
        const float ratioBaselineDepth = baseline/medianDepthKF2;

        if(ratioBaselineDepth<baselineThres)
        {
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

//        map<Point2f,RcKeyPoint,Point2fLess> &keyPoints2 = keyframeKeyPointsMap.at(mnid2);
        SparseMat_<RcHighGradientPoint*> &highGradientPoints2 = keyFrameHighGradientPointsMat.at(mnid2);

//        for(auto &kpit : keyPoints1)
        SparseMatIterator_<RcHighGradientPoint*>
            it = highGradientPoints1.begin(),
            it_end = highGradientPoints1.end();
        for(; it != it_end; ++it)
        {
//            RcKeyPoint &kp1 = *(it.value<RcKeyPoint*>());
            RcHighGradientPoint &kp1 = **it;
//            RcKeyPoint &kp1 = kpit.second;

            if(!kp1.fused /*|| !kp1.intraCheckCount*/)
            {
                continue;
            }

            float tho1 = kp1.tho;

            float x, y;
            if(needRectify)
            {
                // undistort
                cv::Mat mat(1,2,CV_32F);
                mat.at<float>(0,0)=kp1.pt.x;
                mat.at<float>(0,1)=kp1.pt.y;

                mat=mat.reshape(2);
                cv::undistortPoints(mat,mat,pKF->mK,mDistCoef,cv::Mat(), pKF->mK);
                mat=mat.reshape(1);

                x=mat.at<float>(0,0);
                y=mat.at<float>(0,1);
            }
            else
            {
                x = kp1.pt.x;
                y = kp1.pt.y;
            }

            // Xba(p) = K.inv() * xp
            cv::Mat xp1 = (cv::Mat_<float>(1,3) << (x-pKF->cx)*pKF->invfx, (y-pKF->cy)*pKF->invfy, 1.0);
            float invTho = 1.0/tho1;
            cv::Mat D = (Mat_<float>(3,3) << invTho,0,0,0, invTho,0,0,0, invTho);
            Mat KRD = KR*D;

            // project to neighbour keyframe
            Mat xj = KRD*xp1.t() + Kt;
            float tho2 = tho1 / (rjiz.dot(xp1) + tho1 * tjiz);

            // check neighbours
            float uj = xj.at<float>(0) / xj.at<float>(2), vj = xj.at<float>(1) / xj.at<float>(2);
            Point2f disp = Point2f(uj, vj);
            if(needRectify)
            {
                Distort(disp, pKF2);
            }
            uj = disp.x;
            vj = disp.y;

            vector<Point2f> neighbours;
            neighbours.push_back(Point2f(floor(uj), floor(vj)));
            neighbours.push_back(Point2f(floor(uj), ceil(vj)));
            neighbours.push_back(Point2f(ceil(uj), floor(vj)));
            neighbours.push_back(Point2f(ceil(uj), ceil(vj)));
            int validProjectCount = 0;
            for (const Point2f &p: neighbours)
            {/*
                if(keyPoints2.count(p))
                {
                    RcKeyPoint &kp2 = keyPoints2.at(p);*/
                if(highGradientPoints2.ptr(p.y,p.x,false) != NULL)
                {
                    RcHighGradientPoint &kp2 = *highGradientPoints2.ref(p.y,p.x);
                    if(kp2.fused)
//                    if(kp2.fused && !(kp2.interCheckCount > lambdaN))
                    {
                        float tho2n = kp2.tho;
                        float sigma2nSquare = kp2.sigma * kp2.sigma;

                        // Ka test
                        if((tho2 - tho2n) * (tho2 - tho2n) / sigma2nSquare < 3.84)
                        {
                            validProjectCount ++;

                            // project neighbour error factors, list of djn, rjiz*xp, tjiz, sigmajnSquar
                            vector<float> params;
                            params.push_back(1.0f/tho2n);
                            params.push_back(rjiz.dot(xp1));
                            params.push_back(tjiz);
                            params.push_back(sigma2nSquare);

                            vector<vector<float> > &depthErrorEstimateFactor = depthErrorEstimateFactors[kp1.pt];
                            depthErrorEstimateFactor.push_back(params);

                            set<KeyFrame*> &depthErrorKeyframe = depthErrorKeyframes[kp1.pt];
                            depthErrorKeyframe.insert(pKF2);
                        }
                    }
                }
            }
            if(validProjectCount)
            {
                validPoints.push_back(kp1);
            }
        }

        count++;
        if(count==mKN)
        {
            break;
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
    for(RcHighGradientPoint &vkp: validPoints)
    {
//        cout<<"vkp.interCheckCount "<<vkp.interCheckCount<<endl;
        int count = (int)depthErrorKeyframes.at(vkp.pt).size();
        vkp.interCheckCount = count;
        if(count > mLambdaN)
        {
            vector<vector<float>> &params = depthErrorEstimateFactors.at(vkp.pt);
            float dpstar = estimate(params);
//                    cout<<"create estimate "<<dpstar<<" former "<<1.0/vkp.tho<<endl;
                    vkp.tho = 1.0f/dpstar;
            AddPointToMap(vkp, pKF);
        }
        validIndex++;
    }
}


void MapReconstructor::AddPointToMap(RcHighGradientPoint &kp1, KeyFrame* pKF)
{
    if(kp1.intraCheckCount && kp1.interCheckCount)
    {

        float zh = 1.0/kp1.tho;
//        float error = fabs(kp1.mDepth-zh);
        if(zh>0)
        {
//            /if(error >= errorTolerenceFactor * kp1.mDepth /*&& error<0.1 * kp1.mDepth*/)
//            {
////                kp1.mDepth = zh;
//                kp1.tho = (1.0f/(errorTolerenceFactor * kp1.mDepth * kp1.mDepth) + kp1.tho/(kp1.sigma*kp1.sigma))/(1.0f/(errorTolerenceFactor * kp1.mDepth) + 1.0f/(kp1.sigma * kp1.sigma));
//                zh = 1.0f / kp1.tho;
//                //return;
//            }/*else{
//                kp1.mDepth = zh;
//            }*/
//            kp1.mDepth = zh;

            float mTho = 1.0f/kp1.mDepth, mSigma =  (1.0f/(1.0f - initVarienceFactor) / kp1.mDepth - mTho) / 2.0f;
            float diffSqa = (mTho - kp1.tho) * (mTho - kp1.tho);
            float sigmaSqi = mSigma * mSigma, sigmaSqj = kp1.sigma*kp1.sigma;

            if((diffSqa/sigmaSqi + diffSqa/sigmaSqj) < 5.99)
            {
                kp1.tho = (1.0f/sigmaSqi * mTho + 1.0f/sigmaSqj * kp1.tho)/(1.0f/sigmaSqi+ 1.0f/sigmaSqj);
                kp1.mDepth = 1.0f / kp1.tho;
            }
            else
            {
                return;
            }
        }

//        kp1.mDepth = zh;

        cv::Mat x3D = UnprojectStereo(kp1, pKF);
        if(!x3D.empty())
        {
            uchar grey  = kp1.intensity;
            cv::Mat rgb = (cv::Mat_<float>(3,1) <<grey,grey,grey);
            MapPoint* pMP = new MapPoint(x3D,pKF,mpMap,rgb);
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


