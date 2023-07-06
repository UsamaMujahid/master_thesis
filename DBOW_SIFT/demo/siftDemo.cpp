#include <fstream>
#include <iostream>
#include <unistd.h>

#include <chrono>

#include "TemplatedDatabase.h"
#include "TemplatedVocabulary.h"
#include "BowVector.h"
#include "FeatureVector.h"
#include "QueryResults.h"
#include "FORB.h"
#include "FSift.h"
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp> // sift

#include <opencv2/imgproc/imgproc.hpp>

using namespace DBoW2;
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

double mean_feature_detection_t = 0;
double mean_transformation_t = 0;
double mean_image_comparson_t = 0;


/// SIFT Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FSift::TDescriptor, DBoW2::FSift> 
  SiftVocabulary;

/// SIFT Database
typedef DBoW2::TemplatedDatabase<DBoW2::FSift::TDescriptor, DBoW2::FSift> 
  SiftDatabase;

//void loadFeatures(vector<vector<vector<float > > > &features);
//void changeStructure(const cv::Mat &plain, vector<vector<float > > &out);
//void testVocCreation(const vector<vector<vector<float > > > &features);
//void testDatabase(const vector<vector<vector<float > > > &features);

// number of training images
const int NIMAGES = 4;

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

void changeStructure(const cv::Mat &plain, vector<vector<float > > &out)
{
    out.resize(plain.rows);
    for(int j = 0; j < plain.rows; ++j)
    {
        out[j].reserve(plain.cols);
        for(int i = 0; i < plain.cols; i++)
        {
            out[j].push_back(plain.at<float>(j, i));
        }
    }
}

void loadFeatures_old(vector<vector<vector<float > > > &features)
{
    features.clear();
    features.reserve(NIMAGES);
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(1000);
    cout << "Rxtracting SIFT features..." << endl;
    for(int i = 0; i < NIMAGES; ++i)
    {
        stringstream ss;
        ss << "images/image" << i << ".png";

        cv::Mat image = cv::imread(ss.str(), 0);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        sift -> detectAndCompute(image, mask, keypoints, descriptors);
        
        features.push_back(vector<vector<float > >());
        changeStructure(descriptors, features.back());
    }

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

void  loadFeatures(std::vector<string> path_to_images, vector<vector<vector<float > > > &features, ofstream &fs)throw (std::exception){
    //select detector
    features.clear();
    features.reserve(path_to_images.size());
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    cout << "Rxtracting SIFT features..." << endl;

    cv::Mat output;

    double t, sampling_time = 0.033;

    cout << "Extracting   features..." << endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i],0);

        cv::Mat mask;

        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        cout<<"extracting features"<<endl;
        
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        
        sift -> detectAndCompute(image, mask, keypoints, descriptors);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        mean_feature_detection_t =  mean_feature_detection_t + std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        features.push_back(vector<vector<float > >());
        changeStructure(descriptors, features.back());
        
        t = i*sampling_time;
        try
        {
            fs <<keypoints.size()<<","<<i<<","<<t<<"\n";
            fs.flush();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
        

        

        // Draw SIFT keypoints
        /*
        cv::drawKeypoints(image, keypoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cout<<"Number of keypoints"<<keypoints.size()<<endl;
        cv::imshow("Output", output);
        cv::waitKey(0);*/
        

        cout<<"done detecting features"<<endl;

    }
}


void compare_images (std::vector<string> path_to_image, std::vector<string> path_to_image1,vector<vector<vector<float > > > &features)throw (std::exception)
{
        //select detector
    features.clear();
    features.reserve(path_to_image.size());
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    cout << "Rxtracting SIFT features..." << endl;

    cv::Mat output;

    double t, sampling_time = 0.033;


    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cout<<"reading image: "<<path_to_image[0]<<endl;
    cv::Mat image = cv::imread(path_to_image[0]);

    cv::Mat mask;

    if(image.empty())throw std::runtime_error("Could not open image"+path_to_image[0]);
    cout<<"extracting features"<<endl;
    sift -> detectAndCompute(image, mask, keypoints, descriptors);
    features.push_back(vector<vector<float > >());
    changeStructure(descriptors, features.back());
        

    
    cv::Mat output1;



    vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
    cout<<"reading image: "<<path_to_image1[0]<<endl;
    cv::Mat image1 = cv::imread(path_to_image1[0]);

    cv::Mat mask1;

    if(image.empty())throw std::runtime_error("Could not open image"+path_to_image1[0]);
    cout<<"extracting features"<<endl;
    sift -> detectAndCompute(image1, mask1, keypoints1, descriptors1);
    features.push_back(vector<vector<float > >());
    changeStructure(descriptors1, features.back());
        

        

    // Draw SIFT keypoints

   // Matcher - Brute Force
   cv::BFMatcher matcher(cv::NORM_L2);
   std::vector< cv::DMatch > matches;
   matcher.match(descriptors, descriptors1, matches);

    cv::drawMatches(image, keypoints,image1, keypoints1, matches,  output, cv::Scalar::all(-1), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cout<<"Number of keypoints"<<keypoints.size()<<endl;
    cv::imshow("Output", output);
    cv::waitKey(0);

}

void testVocCreation(const vector<vector<vector<float > > > &features)
{
    const int k = 9;
    const int L = 4;
    const WeightingType weight = TF_IDF;
    const ScoringType scoring = L2_NORM;

    SiftVocabulary voc(k, L, weight, scoring);
    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);   
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    for(int i = 0; i < NIMAGES; i++)
    {
        voc.transform(features[i], v1);
        for(int j = 0; j < NIMAGES; j++)
        {
        voc.transform(features[j], v2);
        
        double score = voc.score(v1, v2);
        cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // save the vocabulary to disk
    cout << endl << "Saving sift vocabulary..." << endl;
    voc.save("small_sift_voc.yml.gz");
    cout << "Done" << endl;
}

void testDatabase(const vector<vector<vector<float > > > &features)
{
    cout << "Creating a small database..." << endl;
    SiftVocabulary voc("small_sift_voc.yml.gz");
    SiftDatabase db(voc, false, 0);

    for(int i = 0; i < NIMAGES; i++)
    {
        db.add(features[i]);
    }
    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;
  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
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
  db.save("small_sift_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  SiftDatabase db2("small_sift_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;

}

void testImageInDatabase(const vector<vector<vector<float > > > &features, const vector<vector<vector<float > > > &key_feature,vector<string> &paths, vector<string> &key_path)
{
    // branching factor and depth levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType scoring = L2_NORM;


    int matches = 0;
    const double threshold = 0.1;

    // file pointer
    fstream fout;
  
    // opens an existing csv file or creates a new file.
    fout.open("record_sift.csv", ios::out );
    fout <<"score"<<","<<"index"<<","<<"time"<<","<<"matching"<<"\n";

    cout << "Creating a small database..." << endl;
    //SiftVocabulary voc("small_sift_voc.yml.gz");
    SiftVocabulary voc(k, L, weight, scoring);
    SiftDatabase db(voc, false, 0);


    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    double t, sampling_time = 0.033;

    voc.transform(key_feature[0], v1);

    int counter = 0;


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
        //fout <<score<<","<<j<<","<<t<<"\n";
        //cout << "Image " << 0 << " vs Image " << j << ": " << score << endl;
        /*counter++;
        if (counter = 50)
        {
            voc.transform(features[j], v1);
            counter = 0;
        }*/
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

/*int main()
{
  vector<vector<vector<float > > > features;
  loadFeatures_old(features);
  cout << "Number of Features = " << features.size() << endl;
  testVocCreation(features);

  wait();

  testDatabase(features);

  return 0;
}

*/
// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{
  vector<vector<vector<float > > > features;
  vector<vector<vector<float > > > key_feature;


    // file pointer
    ofstream fout1;

    // opens an existing csv file or creates a new file.
    fout1.open("features_dump_sift.csv", ios::out );
    fout1 <<"num_features"<<","<<"index"<<","<<"time"<<"\n";




  auto images=readImagePaths("images/*.png");
  loadFeatures(images,features,fout1);
  cout << "Number of Features = " << features.size() << endl;


  auto key_image=readImagePaths("base_image/*.png");
  loadFeatures(key_image,key_feature,fout1);
/*
  auto key_image=readImagePaths("comp_image_1/*.png");
  auto key_image1=readImagePaths("comp_image_2/*.png");

  compare_images(key_image,key_image1, features);
*/
  fout1.close();
  //testVocCreation(features);


  //testDatabase(features);

 

  testImageInDatabase(features,key_feature,images, key_image);

    cout << "Average feature detection time[us] = " << mean_feature_detection_t /(images.size() + key_image.size())<<endl;
    cout << "Average Transformation time[us] = " << mean_transformation_t /(images.size() + key_image.size())<<endl;
    cout << "Average image comparison time[us] = " << mean_image_comparson_t /(images.size() + key_image.size())<<endl;

    return 0;
}