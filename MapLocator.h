#pragma once

#include <iostream>  
#include <vector>
#include <cstdlib>
#include <fstream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

const double PI = acos(-1.0);

class MapLocator
{
public:
	MapLocator();
	MapLocator(int _n_boards, int _LL_height, bool _is_log_on = false);
	~MapLocator();

	int n_boards;	// number of boards, should know in advance
	int main_board;		// index of the main board
	int total_iter;
	int current_iter;
	int LL_height;
	bool is_log_on;

	vector<vector<Point2d> > ImagePoints;
	vector<vector<Point2d> > ReprojImagePoints;
	vector<vector<Point3d> > ObjectPoints;
	vector<Size> board_sizes;
	vector<Mat> relative_mat;	// transform matrices from main board to other boards
	Mat main2sphere;	// transform matrix from main board to the sphere's coordinate system (right hand, (0,0,-1) is forward)
	ofstream logFile;

	void initialize();
	void optimize(int max_iter = 100);
	void update();
	double get_temperature();
	Mat compute_Jacobian(const Point3d &p, const Point2d &M, Point2d &projected, double &residual);
	Mat solve_increment(const Mat &Jr, const Mat &r, Mat &x);
	void closeup();
	void visualize(char *imgpath);
	void printMat(const Mat &target);
	void orthogonalize_transform(Mat &mat);
};

