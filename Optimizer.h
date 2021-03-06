#pragma once

#include <iostream>  
#include <vector>
#include <cstdlib>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define MAX_LEN 100
#define WRITE_IMAGE 1
#define CHESSBOARD_SIZE 0.005	// check this parameter with your render scene


using namespace cv;
using namespace std;

struct CamView
{
	vector<vector<Point2d> > ImagePoints;
	vector<vector<Point3d> > ObjectPoints;
	vector<Mat> extrinsic_mat;
	Mat main_extrinsic_mat;		// extrinsic matrix of the main board
	//vector<Point2d> SampledPoints;
};

class Optimizer
{
public:
	Optimizer();
	Optimizer(int _n_boards, Mat _intrinsic_mat, Mat _distortion_mat = Mat::zeros(5, 1, CV_64F));
	~Optimizer();
	
	int n_boards;	// number of boards, should know in advance
	int main_board;		// index of the main board
	int total_iter;
	int current_iter;
	Mat intrinsic_mat;	// camera intrinsic matrix
	Mat distortion_mat;	// camera distortion matrix
	vector<CamView> camera_views;	// holder of all camera view data
	vector<Mat> relative_mat;	// transform matrices from main board to other boards
	vector<Size> board_sizes;

	void initialize();
	void optimize(int max_iter = 100);
	void update_main();
	void update_relative();
	double get_temperature();
	Mat compute_Jacobian_main(const Point3d &p, const Point2d &M, const Mat &main_mat, const Mat &trans_mat, double &residual);
	Mat compute_Jacobian_relative(const Point3d &p, const Point2d &M, const Mat &main_mat, const Mat &trans_mat, double &residual);
	Mat solve_increment(const Mat &Jr, const Mat &r);
	double computeReprojectionErrors(const vector<Point3d>& objectPoints,
		const vector<Point2d>& imagePoints, const Mat& rvec, const Mat& tvec,
		const Mat& cameraMatrix, const Mat& distCoeffs, vector<Point2d>& imagePoints2);
	void drawAxis(Mat &img, const Mat& rvec, const Mat& tvec,
		const Mat& cameraMatrix, const Mat& distCoeffs);
	void closeup();
	void visualize(char *imgpath);
	void printMat(const Mat &target);
	void orthogonalize_transform(Mat &mat);
};

