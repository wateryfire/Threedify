#include <iostream>
#include <vector>
#include <fstream>
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
//using namespace cv;
// - - - - - --- - - - -- - - - - -
/// ORB Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void isInImage(vector<cv::KeyPoint> &keys, float &cx, float &cy, float &rMin, float &rMax, vector<bool> &mask); //for future usage
void createVocabularyFile(ORBVocabulary &voc, std::string &fileName, const vector<vector<cv::Mat> > &features);
void LoadImages(const string &strFile, vector<string> &vstrImageFilenames);
void prepareBagfile (const string & bagFile);
inline vector<cv::Mat> getDescriptorVectorForImg(const string& imgFileName, ORBextractor& extractor)
{
            cv::Mat imageRGB = cv::imread(imgFileName) ;
            cv::Mat imageGray;
            cv::cvtColor(imageRGB, imageGray, CV_RGB2GRAY);
            vector<cv::KeyPoint> keypoints;
            cv::Mat ORBDescriptors;
            extractor(imageGray, cv::Mat(), keypoints, ORBDescriptors);
           
           vector<cv::Mat >  desciptorVector ;
            
            for (int i = 0; i < ORBDescriptors.rows; i++) 
            {

                desciptorVector.push_back(ORBDescriptors.row(i));
               
            }
            return desciptorVector;
    
}
// ----------------------------------------------------------------------------
int main(int argc, const char**  argv)
{
    int nImages = 0;
    vector<vector<cv::Mat > > featuresVector;
    featuresVector.clear();
    std::string vocName;
    int nLevels = 6;
    // Extracting ORB features from bag file
     if ( argc < 3|| argc > 5 || strcmp("-help", argv[1]) == 0 ) //intentionllu to put argv comparison to the end, as there is possibility no argv[1] 
     {
         cout<<"Usage:  "<<endl;
         cout<<"CreateORBVoc -bagfile <bagfile> <VocFilename> (optional):  "<<endl;
         cout<<"CreateORBVoc -imglst <path-to-imagelist> <VocFilename> (optional):  "<<endl;
         return 0;
     }
    

    if (argc== 3 ||  !argv[3] )  // todo: need to add checking conditions that argv[2] contains invalid filename
    {
        vocName = "ORBVoc.txt"; 
      }
     else
    { 
        vocName = string(argv[3]); 
    }
             

    //work  for pic file list
    
    if (strcmp("-imglst", argv[1]) == 0 )
    {
        vector<string> imageFileNames;
        string imgLstFile;
         if ( !argv[2] )  // todo: need to add checking conditions that argv[2] contains invalid filename
        {
            cout << "Invalid ImageList File" << endl;
          }
         else
        { 
            imgLstFile = string(argv[2]); 
        }
       
        LoadImages(imgLstFile, imageFileNames);

        nImages = imageFileNames.size();


        // initialze ORBextractor
        int nLevels = 6;
        //ORBextractor* extractor = new ORBextractor(1000,1.2,nLevels,1,20); //keep the same as in ORBSLAM2 code
        ORBextractor extractor(1000,1.2,nLevels,1,20); //keep the same as in ORBSLAM2 code
      
        featuresVector.reserve(nImages);

       
        cv::Mat imageGray;
        cv::Mat imageRGB;

        
//Extract features from each image from list and organize the features to vector <vector> format
      for(auto &imageit : imageFileNames)
      {
 
            featuresVector.push_back(getDescriptorVectorForImg(imageit, extractor));
              cout << "... Extract ORB for "<<imageit << endl;
      }
    

        cout << "... Extraction done!" << endl;
    }
    else 
    {
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
        
    }
    // Creating the Vocabulary
    // = = =
    // define vocabulary
    const int k = 10; // branching factor
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    ORBVocabulary voc(k, nLevels, weight, score);

    createVocabularyFile(voc, vocName, featuresVector);
    cout << "--- THE END ---" << endl;
    // = = =
    return 0;
}

/**************************************** 
 * for future usage
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
 * *****************************************/
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
            cout << "original String "<<ss<< endl;
            ss >> imageFileName;
            vstrImageFilenames.push_back(imageFileName);
             cout << "... Get Image File Names "<<imageFileName << endl;
        }
    }
}

/********************************************************
 * todo: add configuration file
 * *******************************************************
 * 
 * 
void Initialize(string strSettingsFile)
{
            cv::FileStorage fSettings(strSettingsFile, cv::FileStorage::READ);
            string imageListFile;
            fSettings["ImageList"]>>imageListFile;
            string vocFileName;
             fSettings["VocabularyFile"]>>vocFileName;
             int nfeatures = fSettings["ORBExtractor.nfeatures"];;
             float scaleFactor= fSettings["ORBExtractor.scaleFactors"];
            int scaleLevels=  fSettings["ORBExtractor.scaleLevels"];
            int iniThFAST=  fSettings["ORBExtractor.iniThFAST"];
            int minThFAST=  fSettings["ORBExtractor.minThFAST"];
             int orbVocLevels =  fSettings["ORBVocabulary.levels"];
             int orbVocBranches fSettings["ORBVocabulary.branches"];;
             WeightingType weightingType ;
            fSettings["ORBExtractor.scaleLevels"] >> weightingType ;
            ScoringType scoringType;
            fSettings["ORBExtractor.scaleLevels"] >> scoringType;
}
************************************************************/

