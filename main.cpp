
#include <iostream>  
#include <vector>
#include <omp.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <engine.h>
#include <fstream>
#include "Optimizer.h"
#include "MapLocator.h"

#include <time.h>  

using namespace cv;
using namespace std;


#define UNDISTORT 0
#define DETECT_CORNER 0
#define SOLVECHESS 0
#define SOLVERICOH 1


#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX_LEN 1000
#define SEARCH_SCALE 2.5
float resolution = 1024;

vector<Point3d> generateChessboard3d(int row, int col, double edge)
{
	vector<Point3d> points;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			points.push_back(Point3d((double)j * edge, (double)i * edge, 0.0));
			//cout << (float)j * edge << " " << (float)i * edge << endl;
		}
	}
	return points;
}

double dist3d(Point3d A, Point3d B)
{
	return sqrt((A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y) + (A.z - B.z)*(A.z - B.z));
}

void help()
{
	cout << "Input format:"
		<< "[imagelist path]"
		<< "[chessboard sizes]"
		<< "[envmap path]"
		<< endl;
}

vector<double> getVector(const Mat &_t1f)
{
	Mat t1f;
	_t1f.convertTo(t1f, CV_64F);
	return (vector<double>)(t1f.reshape(1, 1));
}

void printMat(const Mat &target)
{
	int row = target.rows;
	int col = target.cols;
	cout << "[";
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << target.at<double>(i, j) << " ";
		}
		if (i < row - 1) cout << endl;
	}
	cout << "]" << endl;
}

int test(int initial)
{
	int a = initial;
	for (int i = 0; i < 10000000; i++)
	{
		a+=i;
	}
	return a;
}

void testopenmp()
{
	double t1 = (double)getTickCount();
	volatile int temp;
#pragma omp parallel for 
	for (int i = 0; i < 1000; i++)
	{
		temp = test(i);
		//cout << temp << endl;
	}
	t1 = ((double)getTickCount() - t1) / getTickFrequency();
	cout << "Time elapsed: " << t1 << endl;
	exit(1);
}

void refineROI(Rect &rect, Size imgsize)
{
	rect.x = MAX(rect.x, 0);
	rect.y = MAX(rect.y, 0);
	double w = imgsize.width;
	double h = imgsize.height;
	rect.width = MIN(rect.width, w - rect.x);
	rect.height = MIN(rect.height, h - rect.y);
}

int main(int argc, char *argv[])
{
	
	if (argc < 2)
	{
		help();
		return 1;
	}
	//char *matname = argv[1];
	//resolution = atof(argv[2]);
	char *imgpath = argv[1];
	char *sizepath = argv[2];

	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	cameraMatrix.at<double>(0, 0) = 22170.18826229795;//1.634296875 * resolution;
	cameraMatrix.at<double>(1, 1) = 22170.18826229795;//1.634296875 * resolution;
	cameraMatrix.at<double>(0, 2) = 2375.5;//(resolution - 1) / 2;
	cameraMatrix.at<double>(1, 2) = 1583.5;//(resolution - 1) / 2;

	Mat distCoeffs = Mat::zeros(5, 1, CV_64F);


	FILE *fin;
	fin = fopen(sizepath, "r");
	int n_boards = -1;
	vector<Size> patternSizes;
	fscanf(fin, "%d", &n_boards);
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		int w, h;
		fscanf(fin, "%d%d", &w, &h);
		patternSizes.push_back(Size(w, h));
	}
	fclose(fin);
	Optimizer *optimizer = new Optimizer(n_boards, cameraMatrix);
	optimizer->board_sizes = patternSizes;

#if UNDISTORT
	distCoeffs.at<double>(0) = -0.37077668666130514;
	distCoeffs.at<double>(1) = 8.0453921321123243;
	distCoeffs.at<double>(4) = 0.11371622504317971;
	fin = fopen(imgpath, "r");
	int n_pic = -1;
	fscanf(fin, "%d", &n_pic);
	for (int i_view = 0; i_view < n_pic; i_view++)
	{
		char picname[MAX_LEN];
		sprintf(picname, "picture %d", i_view);
		fscanf(fin, "%s", &picname);
		cout << "reading image " << picname << endl;
		Mat view = imread(picname);
		Mat rectified;
		undistort(view, rectified, cameraMatrix, distCoeffs);
		sprintf(picname, "undistort%d.png", i_view);
		cout << "writing " << picname << endl;
		//imshow("preview", view);
		imwrite(picname, rectified);
		cout << "undistorted view " << i_view << ":" << endl;
	}
	fclose(fin);
	exit(1);
#endif

	fin = fopen(imgpath, "r");
	int n_views = -1;
	fscanf(fin, "%d", &n_views);
#if DETECT_CORNER
	FileStorage fs_corners("corners.yml", FileStorage::WRITE);
	vector<Rect> feasible_ROI(n_boards);
	for (int i_view = 0; i_view < n_views; i_view++)
	{
		char picname[MAX_LEN];
		fscanf(fin, "%s", &picname);
		Mat view = imread(picname);
		Mat gray;
		cvtColor(view, gray, CV_BGR2GRAY);
		double t = (double)getTickCount();
//#pragma omp parallel for 
		for (int i_board = 0; i_board < n_boards; i_board++)
		{
			cout << "finding corners for pic " << i_view << " board " << i_board << endl;
			vector<Point2f> corners;
			bool patternfound = false;
			if (i_view > 0 && feasible_ROI[i_board].width > 0)
			{
				Rect ROI = feasible_ROI[i_board];
				ROI.x -= ROI.width * (SEARCH_SCALE - 1) / 2;
				ROI.y -= ROI.height * (SEARCH_SCALE - 1) / 2;
				ROI.width *= SEARCH_SCALE;
				ROI.height *= SEARCH_SCALE;
				refineROI(ROI, gray.size());
				rectangle(view, ROI, Scalar(0, 255, 0));
				patternfound = findChessboardCorners(gray(ROI), patternSizes[i_board], corners);
				for (int i_point = 0; i_point < corners.size(); i_point++)
				{
					corners[i_point] += Point2f(ROI.x, ROI.y);
				}
				if (!patternfound)
				{
					cout << "search ROI failed, running full search" << endl;
				}
			}
			if (!patternfound)
			{
				patternfound = findChessboardCorners(gray, patternSizes[i_board], corners,
					CV_CALIB_CB_ADAPTIVE_THRESH + CV_CALIB_CB_NORMALIZE_IMAGE);
			}

			if (patternfound)
				cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
					TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			
			char store_name[MAX_LEN];
			sprintf(store_name, "view%dboard%d", i_view, i_board);
			fs_corners << store_name << corners;

			int max_w = -1;
			int max_h = -1;
			int min_w = 100000;	// we cannot deal with images that large anyway
			int min_h = 100000;
			for (int i_point = 0; i_point < corners.size(); i_point++)
			{
				max_w = MAX(max_w, corners[i_point].x);
				max_h = MAX(max_h, corners[i_point].y);
				min_w = MIN(min_w, corners[i_point].x);
				min_h = MIN(min_h, corners[i_point].y);
			}
			feasible_ROI[i_board] = Rect(min_w, min_h, max_w - min_w, max_h - min_h);
			rectangle(view, feasible_ROI[i_board], Scalar(255, 0, 0));
			drawChessboardCorners(view, patternSizes[i_board], Mat(corners), patternfound);
		}
		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "Procedure " << i_view << " took " << t << " seconds" << endl;
		sprintf(picname, "pattern%d.png", i_view);
		imwrite(picname, view);
	}
	fs_corners.release();
#else
	cout << "Read corners from file..." << endl;
	FileStorage fs_corners("corners.yml", FileStorage::READ);
	for (int i_view = 0; i_view < n_views; i_view++)
	{
		CamView current_view;
		for (int i_board = 0; i_board < n_boards; i_board++)
		{
			char store_name[MAX_LEN];
			sprintf(store_name, "view%dboard%d", i_view, i_board);
			vector<Point2d> temp;
			fs_corners[store_name] >> temp;
			current_view.ImagePoints.push_back(temp);
			current_view.ObjectPoints.push_back(
				generateChessboard3d(patternSizes[i_board].height, 
					patternSizes[i_board].width, CHESSBOARD_SIZE));
		}
		optimizer->camera_views.push_back(current_view);
	}
	fs_corners.release();
	cout << "Finish reading corners." << endl;
#endif
	fclose(fin);

#if SOLVECHESS
	optimizer->optimize(500);
	optimizer->visualize(imgpath);
	FileStorage fs("camerapos.yml", FileStorage::WRITE);
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		stringstream cur_label;
		cur_label << "Transmat" << i_board;
		fs << cur_label.str() << optimizer->relative_mat[i_board];
	}
	fs.release();
#else
	cout << "Read transform matrices from file..." << endl;
	optimizer->initialize();
	FileStorage fs("camerapos.yml", FileStorage::READ);
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		stringstream cur_label;
		cur_label << "Transmat" << i_board;
		fs[cur_label.str()] >> optimizer->relative_mat[i_board];
	}
	fs.release();
	cout << "Finish reading transform matrices." << endl;

#endif

#if SOLVERICOH	
	char* envmap_path = argv[3];
	Mat envmap = imread(envmap_path);
	Mat gray_map;
	cvtColor(envmap, gray_map, CV_BGR2GRAY);
	MapLocator *RicohLocator = new MapLocator(n_boards, envmap.rows, true);

	RicohLocator->relative_mat = optimizer->relative_mat;
	RicohLocator->board_sizes = patternSizes;
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		cout << "Detecting chessboard " << i_board << endl;
		vector<Point2d> imagePoints;
		vector<Point3d> objectPoints;
		bool patternfound = false;
		//if (i_board != 3)
		patternfound = findChessboardCorners(gray_map, patternSizes[i_board], 
			imagePoints, CV_CALIB_CB_ADAPTIVE_THRESH + CV_CALIB_CB_NORMALIZE_IMAGE);

		objectPoints = generateChessboard3d(patternSizes[i_board].height, 
			patternSizes[i_board].width, CHESSBOARD_SIZE);
		RicohLocator->ObjectPoints.push_back(objectPoints);
		RicohLocator->ImagePoints.push_back(imagePoints);
		if (patternfound)
		{
			cout << "Found chessboard " << i_board << endl;
		}
		else {
			cout << "Not found chessboard " << i_board << endl;
		}
	}
	if (RicohLocator->ObjectPoints.empty())
	{
		cout << "no valid chessboard in the environment map!" << endl;
		exit(1);
	}
	//RicohLocator->mannual();
	RicohLocator->optimize(500);
#endif

	return 0;
}