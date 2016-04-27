#include <iostream>
#include <vector>
#include<fstream>
// DBoW2
//#include <DBoW2ori/DBoW2.h>
//#include <DUtils/DUtils.h>
//#include <DUtilsCV/DUtilsCV.h> // defines macros CVXX
//#include <DVision/DVision.h>
// OpenCV
//#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/nonfree/features2d.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
// ROS
/***********************************************
#include <rosbag/bag.h> 
#include <rosbag/view.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <boost/foreach.hpp>
#include <cv_bridge/cv_bridge.h>
*********************************************** * */
//ORBSLAM2
#include "ORBextractor.h"
#include "ORBVocabulary.h"
using namespace DBoW2;
using namespace DUtils;
using namespace std;
using namespace ORB_SLAM2;
using namespace cv;
// - - - - - --- - - - -- - - - - -
/// ORB Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
ORBVocabulary;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void extractORBFeatures(cv::Mat &image, vector<vector<cv::Mat> > &features, ORBextractor* extractor);
void changeStructureORB( const cv::Mat &descriptor,vector<cv::Mat> &out);
void isInImage(vector<cv::KeyPoint> &keys, float &cx, float &cy, float &rMin, float &rMax, vector<bool> &mask);
void createVocabularyFile(ORBVocabulary &voc, std::string &fileName, const vector<vector<cv::Mat> > &features);
void LoadImages(const string &strFile, vector<string> &vstrImageFilenames);
void prepareBagfile (const string & bagFile);
// ----------------------------------------------------------------------------
int main(int argc, const char**  argv)
{
    int nImages = 0;
    vector<vector<cv::Mat > > features;
    features.clear();
    std::string vociName;
    int nLevels = 6;
    // Extracting ORB features from bag file
     if ( argc < 3|| argc > 5 || strcmp("-help", argv[1]) == 0 ) //intentionllu to put argv comparison to the end, as there is possibility no argv[1] 
     {
         cout<<"Usage:  "<<endl;
         cout<<"CreateORBVoc -bagfile <bagfile> <VocFilename> (optional):  "<<endl;
         cout<<"CreateORBVoc -dir <path> <VocFilename> (optional):  "<<endl;
         return 0;
     }
    

    if (argc== 3 ||  !argv[2] )  // todo: need to add checking conditions that argv[2] contains invalid filename
    {
        vociName = "vociOmni.txt"; 
      }
     else
    { 
        vociName = string(argv[2]); 
    }
             
// = = =
// load bag file
//=========================================
//todo: support bagfile in future
/*******************************************************************************************
    if (strcmp("-bagfile", argv[1]) == 0 )
    {
        string bagFile = "/path/to/your/bag/file"; //todo: input file should be provided via input line
        rosbag::Bag bag(bagFile);
        rosbag::View viewTopic(bag, rosbag::TopicQuery("/camera/image_raw"));
        nImages = viewTopic.size();


        // initialze ORBextractor

        ORBextractor* extractor = new ORBextractor(1000,1.2,nLevels,1,20); //keep the same as in ORBSLAM2 code


        features.reserve(nImages);

        cv_bridge::CvImageConstPtr cv_ptr;
        cv::Mat image;
        cout << "> Using bag file: " << bagFile << endl;
        cout << "> Extracting Features from " << nImages << " images..." << endl;
        
        BOOST_FOREACH(rosbag::MessageInstance const m, viewTopic)
        {
            sensor_msgs::Image::ConstPtr i = m.instantiate<sensor_msgs::Image>();
            if (i != NULL) {
                cv_ptr = cv_bridge::toCvShare(i);
                cvtColor(cv_ptr->image, image, CV_RGB2GRAY);
                extractORBFeatures(image, features, extractor);
            }
        }
    
        bag.close();
        cout << "... Extraction done!" << endl;
    }
    ----------------------------------------------------------------------------------------------------------------------*/
    //work  for pic file list
    
    if (strcmp("-dir", argv[1]) == 0 )
    {
    vector<string> imageFileNames;

    string strFile = string(argv[2])+"/rgb.txt";
    LoadImages(strFile, imageFileNames);

    int nImages = imageFileNames.size();


        // initialze ORBextractor
        int nLevels = 6;
        ORBextractor* extractor = new ORBextractor(1000,1.2,nLevels,1,20); //keep the same as in ORBSLAM2 code

        vector<vector<cv::Mat > > features;
        features.clear();
        features.reserve(nImages);

       
        cv::Mat imageGray;
         cv::Mat imageRGB;

        
//todo:  read from files
      for(auto &imageit : imageFileNames)
      {
                imageRGB = imread(imageit) ;
                cvtColor(imageRGB, imageGray, CV_RGB2GRAY);
                extractORBFeatures(imageGray, features, extractor);
            
        }
    

        cout << "... Extraction done!" << endl;
    }
    else 
    {
        
    }
    // Creating the Vocabulary
    // = = =
    // define vocabulary
    const int k = 10; // branching factor
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    ORBVocabulary voc(k, nLevels, weight, score);
    createVocabularyFile(voc, vociName, features);
    cout << "--- THE END ---" << endl;
    // = = =
    return 0;
}
// ----------------------------------------------------------------------------
void extractORBFeatures(cv::Mat &image, vector<vector<cv::Mat> > &features, ORBextractor* extractor) {
vector<cv::KeyPoint> keypoints;
cv::Mat descriptorORB;
// extract
(*extractor)(image, cv::Mat(), keypoints, descriptorORB);
//this need to check, personally, I dont think this needed for cluster.
// reject features outside region of interest
//vector<bool> mask;
//float cx = 318.311759; float cy = 243.199269;
//float rMin = 50; float rMax = 240;
//isInImage(keypoints, cx, cy, rMin, rMax, mask);
// create descriptor vector for the vocabulary
features.push_back(vector<cv::Mat>());
changeStructureORB(descriptorORB,  features.back());
}
// ----------------------------------------------------------------------------
void changeStructureORB( const cv::Mat &descriptor, vector<cv::Mat> &out) {
for (int i = 0; i < descriptor.rows; i++) {
//no mask is needed
//if(mask[i]) {
out.push_back(descriptor.row(i));
//}
}
}
// ----------------------------------------------------------------------------
void isInImage(vector<cv::KeyPoint> &keys, float &cx, float &cy, float &rMin, float &rMax, vector<bool> &mask) {
int N = keys.size();
mask = vector<bool>(N, false);
for(int i=0; i<N; i++) {
cv::KeyPoint kp = keys[i];
float uc = (kp.pt.x-cx);
float vc = (kp.pt.y-cy);
float rho = sqrt(uc*uc+vc*vc);
if(rho>=rMin && rho<=rMax) {
mask[i] = true;
}
}
}
// ----------------------------------------------------------------------------
void createVocabularyFile(ORBVocabulary &voc, std::string &fileName, const vector<vector<cv::Mat> > &features)
{
cout << "> Creating vocabulary. May take some time ..." << endl;
voc.create(features);
cout << "... done!" << endl;
cout << "> Vocabulary information: " << endl
<< voc << endl << endl;
// save the vocabulary to disk
cout << endl << "> Saving vocabulary..." << endl;
voc.saveToTextFile(fileName);
cout << "... saved to file: " << fileName << endl;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames)
{
    ifstream f;
    f.open(strFile.c_str());

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            string imageFileName;
            ss >> imageFileName;
            vstrImageFilenames.push_back(imageFileName);
        }
    }
}