#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    string path_data = "../data";
    string associate_data = path_data + "/associate.txt";

    ifstream myfile;
    myfile.open(associate_data,std::ios_base::in);
    if (myfile.is_open())
    {
        cout<<"successed"<<endl;
        string rgb_time,rgb_file,depth_time,depth_file;
        cv::Mat color,depth,last_color;
        list<cv::Point2f> keypoints;
        for (int index = 0; index <100 ; ++index)
        {
            myfile>>rgb_time>>rgb_file>>depth_time>>depth_file;
            color = cv::imread(path_data+"/"+rgb_file);
            depth = cv::imread(path_data+"/"+depth_file,-1);

            if (index == 0)
            {
                vector<cv::KeyPoint> kps;
                cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
                detector->detect(color,kps);

                for (auto kp:kps) {
                    keypoints.push_back(kp.pt);
                }

                last_color = color;
            }

            if (color.data == nullptr || depth.data == nullptr)
                continue;

            vector<cv::Point2f> next_keypoints;
            vector<cv::Point2f> last_keypoints;
            for (auto kp:keypoints)
            {
                last_keypoints.push_back(kp);
            }

            vector<unsigned char > status;
            vector<float> error;

            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

            cout<<"this is "<<index<<" tests"<<endl;
            cv::calcOpticalFlowPyrLK(last_color,color,last_keypoints,next_keypoints,status,error);

            chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
            cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

            int  i = 0;
            for (auto iter = keypoints.begin();iter!= keypoints.end();i++)
            {
                if (status[i] == 0)
                {
                    iter = keypoints.erase(iter);
                    continue;
                }
                *iter = next_keypoints[i];
                iter++;
            }

            cout<<"tracked keypoints: "<<keypoints.size()<<endl;
            if (keypoints.size() == 0)
            {
                cout<<"all keypoints are lost."<<endl;
                break;
            }
            // 画出 keypoints
            cv::Mat img_show = color.clone();
            for ( auto kp:keypoints )
                cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
            cv::imshow("corners", img_show);
            cv::waitKey(0);
            last_color = color;

        }

    }

    return 0;
}