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

public:

	MapReconstructor(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc, Tracking* pTracker);

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


private:

	//Extract and store the edge profile info for a key frame
	void ExtractEdgeProfile(KeyFrame *pKeyFrame);

    Map* mpMap;
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;
    Tracking* mpTracker;

    std::list<KeyFrame*> mlpKeyFrameQueue;
    std::mutex mMutexForKeyFrameQueue;

	//To indicate the current status of jobs.
	JobStatus mStatus_KeyFrameQueueProcess;
	JobStatus mStatus_RealTimeMapReconstruction;
	JobStatus mStatus_FullMapReconstruction;

};

} //namespace ORB_SLAM2


#endif /* INCLUDE_MAPRECONSTRUCTOR_H_ */
