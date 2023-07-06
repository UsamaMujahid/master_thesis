/**
 * Date:  2016
 * Author: Rafael Muñoz Salinas
 * Description: demo application of DBoW3
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <chrono>

// DBoW3
#include "DBoW3.h"

//Glob
#include <glob.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#ifdef USE_CONTRIB 
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"

using namespace DBoW3;
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


double mean_feature_detection_t = 0;
double mean_transformation_t = 0;
double mean_image_comparson_t = 0;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}


vector<string> readImagePaths(string target_location){
    vector<cv::String> fn;
    vector<string> paths;

    cv::String path(target_location);

    cv::glob(path, fn, true);
    size_t count = fn.size();
    cout <<"Total number of files found in the folder = "<<count;
    for(int i=0;i<count;i++)    paths.push_back(fn[i]);
        return paths;
}




vector< cv::Mat  >  loadFeatures( fstream &fs, std::vector<string> path_to_images,string descriptor="") throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create();
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;

    cv::Mat output;
    double t, sampling_time = 0.033;
    cout << "Extracting   features..." << endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i],0);
        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        cout<<"extracting features"<<endl;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    mean_feature_detection_t =  mean_feature_detection_t + std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        features.push_back(descriptors);


        // Draw keypoints
      /*cv::drawKeypoints(image, keypoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cout<<"Number of keypoints"<<keypoints.size()<<endl;
        cv::imshow("Output", output);
        cv::waitKey(0);*/

        t = i*sampling_time;
        fs <<keypoints.size()<<","<<i<<","<<t<<"\n";
        cout<<"done detecting features"<<endl;
    }
    return features;
}


vector< cv::Mat  >  compare_images( std::vector<string> path_to_image,std::vector<string> path_to_image1,string descriptor="") throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create(30);
    else if (descriptor=="brisk") fdetector=cv::BRISK::create(30);
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create(30);
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;

    cv::Mat output;
    double t, sampling_time = 0.033;
    cout << "Extracting   features..." << endl;

    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cout<<"reading image: "<<path_to_image[0]<<endl;
    cv::Mat image = cv::imread(path_to_image[0],0);
    if(image.empty())throw std::runtime_error("Could not open image"+path_to_image[0]);
    cout<<"extracting features"<<endl;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    mean_feature_detection_t =  mean_feature_detection_t + std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    features.push_back(descriptors);


    vector<cv::Mat>    features1;

    cv::Mat output1;
    cout << "Extracting   features..." << endl;

    vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
    cout<<"reading image: "<<path_to_image1[0]<<endl;
    cv::Mat image1 = cv::imread(path_to_image1[0]);
    if(image.empty())throw std::runtime_error("Could not open image"+path_to_image1[0]);
    cout<<"extracting features"<<endl;
    fdetector->detectAndCompute(image1, cv::Mat(), keypoints1, descriptors1);
    features1.push_back(descriptors1);


   // Matcher - Brute Force
   cv::BFMatcher matcher(cv::NORM_L2);
   std::vector< cv::DMatch > matches;
   matcher.match(descriptors, descriptors1, matches);


   //-- Quick calculation of max and min distances between Keypoints
    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors1.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    } 

   std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors.rows; i++ )
    { if( matches[i].distance <= max(2*min_dist, 0.08) )
      { good_matches.push_back( matches[i]); }
    }

    cv::drawMatches(image, keypoints,image1, keypoints1, matches,  output, cv::Scalar::all(-1), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cout<<"Number of keypoints"<<keypoints.size()<<endl;
    cv::imshow("Output", output);
    cv::waitKey(0);


    
    return features;
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<cv::Mat> &features)
{
    // branching factor and depth levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L2_NORM;

    DBoW3::Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    for(size_t i = 0; i < features.size(); i++)
    {
        voc.transform(features[i], v1);
        for(size_t j = 0; j < features.size(); j++)
        {
            voc.transform(features[j], v2);

            double score = voc.score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;
}



// ----------------------------------------------------------------------------

void testImageInDatabase(const vector<cv::Mat> &features, const vector<cv::Mat> &key_feature,vector<string> &paths, vector<string> &key_path)
{
    // branching factor and depth levels
    const int k = 10;
    const int L = 5;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L2_NORM;

    int matches = 0;
    const double threshold = 0.1;

    // file pointer
    fstream fout;
  
    // opens an existing csv file or creates a new file.
    fout.open("record.csv", ios::out );
    fout <<"score"<<","<<"index"<<","<<"time"<<","<<"matching"<<"\n";

    DBoW3::Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    //cout << "Vocabulary information: " << endl
    //     << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    double t, sampling_time = 0.033;

    voc.transform(key_feature[0], v1);
    int counter = 0;
    cout <<"Feature size = "<<features.size()<<endl;
    for(size_t j = 0; j < features.size(); j++)
    {

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        voc.transform(features[j], v2);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        mean_transformation_t =  mean_transformation_t + std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        double score = abs(voc.score(v1, v2));
        end = std::chrono::steady_clock::now();
        mean_image_comparson_t =  mean_image_comparson_t + std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        t = j*sampling_time;
        
        //cout << "Image " << 0 << " vs Image " << j << ": " << score << endl;
        //counter++;

        fout <<score<<","<<j<<","<<t<<","<<matches<<"\n";
/*
        if ((score < threshold) & ((j+1)<features.size()))
        {
            voc.transform(features[j+1], v1);
            fout <<score<<","<<j<<","<<t<<","<<matches<<"\n";
            matches = 0;
        //    counter = 0;
        }
        else
        {
            matches++;
            fout <<score<<","<<j<<","<<t<<","<<0<<"\n";
            
        }
        */
    }


    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;
}

////// ----------------------------------------------------------------------------

void testDatabase(const  vector<cv::Mat > &features)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    Vocabulary voc("small_voc.yml.gz");

    Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Database db2("small_db.yml.gz");
    cout << "... done! This is: " << endl << db2 << endl;
}


// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc<=1){
            cerr<<"Usage:  descriptor_name     image0 image1 ... \n\t descriptors:brisk,surf,orb ,akaze(only if using opencv 3)"<<endl;
             return -1;
        }

        string descriptor=argv[1];

        // file pointer
        fstream fout1;
    
        // opens an existing csv file or creates a new file.
        
        fout1.open("features_dump.csv", ios::out );
        fout1 <<"num_features"<<","<<"index"<<","<<"time"<<"\n";




        auto images=readImagePaths("images/*.png");
        vector< cv::Mat   >   features= loadFeatures(fout1,images,descriptor);

        auto key_image=readImagePaths("base_image/*.png");
        vector< cv::Mat   >   key_feature= loadFeatures(fout1,key_image,descriptor);

        

        /* auto key_image=readImagePaths("comp_image_1/*.png");
         auto key_image1=readImagePaths("comp_image_2/*.png");

         vector< cv::Mat   >   feature= compare_images(key_image,key_image1, descriptor);
*/

        
        //testVocCreation(features);


        //testDatabase(features);

        

        testImageInDatabase(features,key_feature,images, key_image);

    cout << "Average feature detection time[us] = " << mean_feature_detection_t /(images.size() + key_image.size())<<endl;
    cout << "Average Transformation time[us] = " << mean_transformation_t /(images.size() + key_image.size())<<endl;
    cout << "Average image comparison time[us] = " << mean_image_comparson_t /(images.size() + key_image.size())<<endl;
    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
