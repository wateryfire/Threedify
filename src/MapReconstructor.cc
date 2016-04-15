/*
 * MapReconstructor.cc
 *
 *  Created on: Apr 11, 2016
 *      Author: qiancan
 */

#include "MapReconstructor.h"

namespace ORB_SLAM2
{

MapReconstructor::MapReconstructor(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc, Tracking* pTracker):
		mpMap(pMap), mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpTracker(pTracker)
{
	mStatus_KeyFrameQueueProcess=INITIALIZED;
	mStatus_RealTimeMapReconstruction=INITIALIZED;
	mStatus_FullMapReconstruction=INITIALIZED;
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
		usleep(3000);

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

    // TODO: Remove the sleep process, once the real code is implemented.
    usleep(2000);
}

void MapReconstructor::RunToReconstructMap()
{
	while(mStatus_RealTimeMapReconstruction!=STARTED)
	{
		usleep(3000);
	}

	cout << "MapReconstructor: Start thread execution for map reconstruction during SLAM tracking." << endl;

	while(mStatus_RealTimeMapReconstruction!=STOPPED)
	{
		// TODO: Remove the sleep process, once the real code is implemented.
		usleep(10000);
	}

	cout << "MapReconstructor: End thread execution for map reconstruction during SLAM tracking." << endl;


	/*while(this->mStatus_FullMapReconstruction!=STARTED)
	{
		usleep(3000);
	}*/

	cout << "MapReconstructor: Start thread execution for full map reconstruction." << endl;

	// TODO: Remove the sleep process, once the real code is implemented.
	usleep(10000);

	cout << "MapReconstructor: End thread execution for full map reconstruction." << endl;

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


