#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

enum {SURF_D = 0, SIFT_D, ORB_D, AKAZE_D, BRISK_D}; // detectors
enum {FLANN_M = 0, BRUTEFORCE_M, BRUTEFORCE_HUMMING_M}; // matchers

// CHOOSE detector and matcher algorithm
int DETECTOR = ORB_D;
int MATCHER = FLANN_M;

int main()
{
    // 1. load images
    std::string dataDirPath = "../data/";
    std::string path1 = dataDirPath + "table_1.jpg";
    std::string path2 = dataDirPath + "table_2.jpg";
    std::string path3 = dataDirPath + "table_3.jpg";

    cv::Mat imgLeft = cv::imread(path2);
    cv::Mat imgRight = cv::imread(path3);

    cv::cvtColor( imgLeft, imgLeft, cv::COLOR_BGR2GRAY );
    cv::cvtColor( imgRight, imgRight, cv::COLOR_BGR2GRAY );
    cv::equalizeHist( imgLeft, imgLeft);
    cv::equalizeHist( imgRight, imgRight);

    // 2. find matching keypoints
    for(int detIdx = 1; detIdx < 5; detIdx++) {
        for(int matIdx = 0; matIdx < 3; matIdx++) {

            std::string resultPath = dataDirPath;
            DETECTOR = detIdx;
            MATCHER = matIdx;

            // 2.1 detect key points
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            cv::Mat descriptors1, descriptors2;

            switch (DETECTOR) {
            case SURF_D :
            {
                int minHessian = 400;
                cv::Ptr<cv::xfeatures2d::SURF> surf_detector = cv::xfeatures2d::SURF::create( minHessian );
                surf_detector->detectAndCompute( imgRight, cv::noArray(), keypoints1, descriptors1 );
                surf_detector->detectAndCompute( imgLeft, cv::noArray(), keypoints2, descriptors2 );
                std::cout << "-- Detector : SURF" << std::endl;
                resultPath = resultPath + "SURF_";
                break;
            }
            case AKAZE_D:
            {
                cv::Ptr<cv::AKAZE> akaze_detector = cv::AKAZE::create();
                akaze_detector->detectAndCompute( imgRight, cv::noArray(), keypoints1, descriptors1 );
                akaze_detector->detectAndCompute( imgLeft, cv::noArray(), keypoints2, descriptors2 );
                std::cout << "-- Detector : AKAZE" << std::endl;
                resultPath = resultPath + "AKAZE_";
                break;
            }
            case SIFT_D :
            {
                cv::Ptr<cv::Feature2D> sift_detector = cv::SIFT::create();
                sift_detector->detectAndCompute( imgRight, cv::noArray(), keypoints1, descriptors1 );
                sift_detector->detectAndCompute( imgLeft, cv::noArray(), keypoints2, descriptors2 );
                std::cout << "-- Detector : SIFT" << std::endl;
                resultPath = resultPath + "SIFT_";
                break;
            }
            case ORB_D :
            {
                cv::Ptr<cv::Feature2D> orb_detector = cv::ORB::create();
                orb_detector->detectAndCompute( imgRight, cv::noArray(), keypoints1, descriptors1 );
                orb_detector->detectAndCompute( imgLeft, cv::noArray(), keypoints2, descriptors2 );
                std::cout << "-- Detector : ORB" << std::endl;
                resultPath = resultPath + "ORB_";
                break;
            }
            case BRISK_D:
                cv::Ptr<cv::Feature2D> brisk_detector = cv::BRISK::create();
                brisk_detector->detectAndCompute( imgRight, cv::noArray(), keypoints1, descriptors1 );
                brisk_detector->detectAndCompute( imgLeft, cv::noArray(), keypoints2, descriptors2 );
                std::cout << "-- Detector : BRISK" << std::endl;
                resultPath = resultPath + "BRISK_";
                break;
            }

            // 2.2 Matching descriptor vectors with a bruteForce matcher or flannBased matcher

            std::vector< cv::DMatch > good_matches;
            float ratio_thresh = 0.3;

            if (! descriptors1.empty() && ! descriptors2.empty() ) {
                switch (MATCHER) {
                case BRUTEFORCE_M :
                {
                    std::vector< cv::DMatch > matches;
                    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
                    matcher->match( descriptors1, descriptors2, matches );
                    // Quick calculation of min-max distances between keypoints
                    double max_dist = 0;
                    double min_dist = 100;
                    for(int i = 0; i < descriptors1.rows ; i++) {
                        double dist = matches[i].distance;
                        if( dist < min_dist ) min_dist = dist;
                        if( dist > max_dist ) max_dist = dist;
                    }
                    // Use only "good" matches (i.e. whose distance is less than 2 X min_dist )
                    for(int i = 0 ; i < descriptors1.rows ; i++) {
                        if( matches[i].distance < 2*min_dist ) {
                            good_matches.push_back( matches[i] );
                        }
                    }
                    std::cout << "-- Matcher : BRUTEFORCE" << std::endl;
                    resultPath = resultPath + "BRUTEFORCE.bmp";
                    break;
                }
                case FLANN_M :
                {
                    descriptors1.convertTo(descriptors1, CV_32F);
                    descriptors2.convertTo(descriptors2, CV_32F);

                    std::vector< std::vector<cv::DMatch> > knn_matches;
                    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
                    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
                    // Filter matches using the Lowe's ratio test
                    for (size_t i = 0; i < knn_matches.size(); i++) {
                        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                            good_matches.push_back(knn_matches[i][0]);
                        }
                    }
                    std::cout << "-- Matcher : FLANN" << std::endl;
                    resultPath = resultPath + "FLANN.bmp";
                    break;
                }
                case BRUTEFORCE_HUMMING_M :
                {
                    descriptors1.convertTo(descriptors1, CV_8U);
                    descriptors2.convertTo(descriptors2, CV_8U); // ???

                    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
                    std::vector< std::vector<cv::DMatch> > knn_matches;
                    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2);
                    for (size_t i = 0; i < knn_matches.size(); i++) {
                        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                            good_matches.push_back(knn_matches[i][0]);
                        }
                    }
                    std::cout << "-- Matcher : BRUTEFORCE with norm huming" << std::endl;
                    resultPath = resultPath + "BRUTEFORCE_HUMMING.bmp";
                    break;
                }
                }
            }


            // 2.3 save result
            cv::Mat detectedMatches;
            cv::drawMatches( imgRight, keypoints1, imgLeft, keypoints2, good_matches, detectedMatches, cv::Scalar::all(-1),
                         cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            cv::imwrite(resultPath, detectedMatches);
        }
    }

    return 0;
}
