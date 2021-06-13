#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>
#include <string>

#include <opencv2/opencv.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;
using namespace cv;
using namespace g2o;
using namespace Eigen;

struct Measurement1
{
    Measurement1(Eigen::Vector3d p, float g):p_world(p),grayvalue(g){}
    Eigen::Vector3d p_world;
    float  grayvalue;
};

inline Eigen::Vector3d project2Dto3D(int x,int y,int d, float fx, float fy, float cx, float cy, float dep_sca)
{
    float zz = float(d)/dep_sca;
    float xx = (x-cx)*zz/fx;
    float yy = (y-cy)*zz/fy;
    return Eigen::Vector3d(xx,yy,zz);
}

inline Eigen::Vector2d project3Dto2D(float x, float y, float z, float fx, float fy, float cx, float cy)
{
    double uu = fx*x/z + cx;
    double vv = fy*y/z + cy;
    return Eigen::Vector2d(uu,vv);
}

bool poseEsitimationDirect(const vector<Measurement1>& measurements,cv::Mat* gray,Eigen::Matrix3f& K,Eigen::Isometry3d& Tcw);

class EdgeSE3ProjectDirect: public BaseUnaryEdge<1,double ,VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSE3ProjectDirect(){}

    EdgeSE3ProjectDirect(Eigen::Vector3d pw,cv::Mat* gray,float fx,float fy,float cx,float cy):p_w(pw),img(gray),f_x(fx),f_y(fy),c_x(cx),c_y(cy) {}

    virtual  void computeError(){
        const VertexSE3Expmap* v = static_cast<VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d p_trans = v->estimate().map(p_w);
        Eigen::Vector2d p2_uv = project3Dto2D(p_trans[0],p_trans[1],p_trans[2],f_x,f_y,c_x,c_y);
        if (p2_uv[0]<4||(p2_uv[0]+4)>img->cols||p2_uv[1]<4||(p2_uv[1]+4)>img->rows)
        {
            _error(0,0) = 0.0;
            this->setLevel(1);//不优化
        } else
        {
            _error(0,0) = getPixelValue(p2_uv[0],p2_uv[1]) - _measurement;
        }

    }

    virtual void linearizeOplus()
    {
        if (level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double ,1,6>::Zero();
            return;
        }
        VertexSE3Expmap* v1 = static_cast<VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d xyz_trans = v1->estimate().map(p_w);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz*invz;

        double u = f_x*x*invz+c_x;
        double v = f_y*y*invz+c_y;

        Eigen::Matrix<double,2,6> jacobian_uv;
        jacobian_uv ( 0,0 ) = - x*y*invz_2 *f_x;
        jacobian_uv ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *f_x;
        jacobian_uv ( 0,2 ) = - y*invz *f_y;
        jacobian_uv ( 0,3 ) = invz *f_x;
        jacobian_uv ( 0,4 ) = 0;
        jacobian_uv ( 0,5 ) = -x*invz_2 *f_x;

        jacobian_uv ( 1,0 ) = - ( 1+y*y*invz_2 ) *f_y;
        jacobian_uv ( 1,1 ) = x*y*invz_2 *f_y;
        jacobian_uv ( 1,2 ) = x*invz *f_y;
        jacobian_uv ( 1,3 ) = 0;
        jacobian_uv ( 1,4 ) = invz *f_y;
        jacobian_uv ( 1,5 ) = -y*invz_2 *f_y;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
        jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

        _jacobianOplusXi = jacobian_pixel_uv*jacobian_uv;

    }

    virtual bool read ( std::istream& in ) {}
    virtual bool write ( std::ostream& out ) const {}

protected:
    inline float getPixelValue(float x,float y)
    {
        uchar* data = &img->data[int(y)*img->step+int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);

        return float(
                (1-xx)*(1-yy)*data[0]+
                xx*(1-yy)*data[1]+
                yy*(1-xx)*data[img->step]+
                xx*yy*data[img->step+1]
        );
    }

public:
    Eigen::Vector3d p_w;
    float f_x = 0,f_y = 0,c_x = 0,c_y = 0;
    cv::Mat* img = nullptr;
};





int main()
{
    //srand ( ( unsigned int ) time ( 0 ) );
    string path_data = "../data";
    string associate_data = path_data + "/associate.txt";

    ifstream myfile;
    myfile.open(associate_data,std::ios_base::in);
    if (myfile.is_open())
    {
        cout<<"successed"<<endl;
        string rgb_time,rgb_file,depth_time,depth_file;
        cv::Mat color,depth,gray;

        vector<Measurement1> measurement;
        float cx = 325.5;
        float cy = 253.5;
        float fx = 518.0;
        float fy = 519.0;

        float depth_scale = 1000.0;
        Eigen::Matrix3f K;
        K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;

        Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
        cv::Mat pre_color;

        for (int index = 0; index <10 ; ++index)
        {
            cout<<"**********  loop  "<<index<<"*********"<<endl;
            myfile >> rgb_time >> rgb_file >> depth_time >> depth_file;
            color = cv::imread(path_data + "/" + rgb_file);
            depth = cv::imread(path_data + "/" + depth_file, -1);
            if(color.data == 0 ||depth.data == 0)
                continue;
            cv::cvtColor(color,gray,cv::COLOR_BGR2GRAY);

            if (index == 0)
            {
                vector<cv::KeyPoint> kps;
                cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
                detector->detect(color, kps);

                for (auto kp:kps)
                {
                    if (kp.pt.x<20||kp.pt.y<20||(kp.pt.x+20)>color.cols || (kp.pt.y+20)>color.rows)
                        continue;

                    ushort d = depth.ptr<ushort >(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
                    if (d == 0)
                        continue;

                    Eigen::Vector3d p3d = project2Dto3D(kp.pt.x,kp.pt.y,d,fx,fy,cx,cy,depth_scale);
                    float grayscale = float(gray.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)]);

                    measurement.push_back(Measurement1(p3d,grayscale));
                }

                pre_color = color.clone();
                continue;
            }
            //chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            poseEsitimationDirect(measurement,&gray,K,Tcw);
            //chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
            //cout<<"direct method costs time: "<<time_used.count() <<" seconds."<<endl;
            cout<<"Tcw="<<Tcw.matrix() <<endl;

            // plot the feature points
            cv::Mat img_show ( color.rows*2, color.cols, CV_8UC3 );
            pre_color.copyTo ( img_show ( cv::Rect ( 0,0,color.cols, color.rows ) ) );
            color.copyTo ( img_show ( cv::Rect ( 0,color.rows,color.cols, color.rows ) ) );
            for ( Measurement1 m:measurement )
            {
                if ( rand() > RAND_MAX/5 )
                    continue;
                Eigen::Vector3d p = m.p_world;
                Eigen::Vector2d pixel_prev = project3Dto2D ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
                Eigen::Vector3d p2 = Tcw*m.p_world;
                Eigen::Vector2d pixel_now = project3Dto2D ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
                if ( pixel_now(0,0)<0 || pixel_now(0,0)>=color.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=color.rows )
                    continue;

                float b = 255*float ( rand() ) /RAND_MAX;
                float g = 255*float ( rand() ) /RAND_MAX;
                float r = 255*float ( rand() ) /RAND_MAX;
                cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 8, cv::Scalar ( b,g,r ), 2 );
                cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), 8, cv::Scalar ( b,g,r ), 2 );
                cv::line ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), cv::Scalar ( b,g,r ), 1 );
            }
            cv::imshow ( "result", img_show );
            cv::waitKey ( 0 );

        }

    }
    myfile.close();

    return 0;
}


bool poseEsitimationDirect(const vector<Measurement1>& measurements,cv::Mat* gray,Eigen::Matrix3f& K,Eigen::Isometry3d& Tcw)
{

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;
    /*std::unique_ptr<DirectBlock::LinearSolverType> linearSolver ( new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>() );
    std::unique_ptr<DirectBlock> solver_ptr (new DirectBlock( std::move(linearSolver) ));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr) );
    */
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock* solver_ptr = new DirectBlock(unique_ptr<DirectBlock::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<DirectBlock>(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);


    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(),Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    int id = 1;
    for (auto m:measurements)
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect(m.p_world,gray,K(0,0),K(1,1),K(0,2),K(1,2));
        edge->setVertex(0,pose);
        edge->setMeasurement(m.grayvalue);
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity());
        edge->setId(id++);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        rk->setDelta(1.0);
        optimizer.addEdge(edge);

    }
    cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    Tcw = pose->estimate();

}

