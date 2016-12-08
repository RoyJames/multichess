
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

#define SOLVECHESS 0
#define SOLVERICOH 0
#define UNDISTORT 0

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX_LEN 1000
#define SEARCH_SCALE 2.5
const double chessboard_size = 0.005;	// check this parameter with your render scene
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
		<< "[name of the mat file (including .mat)]"
		<< "[list of picture paths]"
		<< "[number of chessboards used]"
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

	FileStorage fs_corners("corners.yml", FileStorage::WRITE);

	Optimizer *optimizer = new Optimizer(n_boards, cameraMatrix);
	vector<Rect> feasible_ROI(n_boards);
	fin = fopen(imgpath, "r");
	int n_views = -1;
	fscanf(fin, "%d", &n_views);
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
				patternfound = findChessboardCorners(gray, patternSizes[i_board], corners);
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
	exit(1);

	/*
	Engine *ep; 
	if (!(ep = engOpen(NULL))) 
	{
		cout << "Can't start Matlab engine!" << endl;
		return 1;
	}
	engEvalString(ep, "clear");
	cout << "engine started!" << endl;
	engEvalString(ep, "addpath('D:\\codes\\libcbdetect')");
	char command[MAX_LEN];
	sprintf(command, "load('%s');", matname);
	engEvalString(ep, command);

	mxArray *chessboards = engGetVariable(ep, "chessboards");
	mxArray *corners = engGetVariable(ep, "corners");
	mxArray *matching = engGetVariable(ep, "matching");
	double *matching_table = mxGetPr(matching);

	int n = mxGetN(chessboards);
	cout << "Total picture count: " << n << endl;
	*/
	// we have only one camera with known intrinsics


#if UNDISTORT
	distCoeffs.at<double>(0) = -0.37077668666130514;
	distCoeffs.at<double>(1) = 8.0453921321123243;
	distCoeffs.at<double>(4) = 0.11371622504317971;
	FILE *fin;
	fin = fopen(imgpath, "r");	
	for (int i_view = 0; i_view < 11; i_view++)
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
	exit(1);
#endif
	/*
	vector<vector<Point3d> > camera_buffer;
	vector<vector<Mat> > camera_extrinsics;

	vector<vector<Mat> > tvecs_all;
	vector<vector<Mat> > rvecs_all;
	vector<vector<vector<Point2d> > > imagePoints_all;
	vector<vector<vector<Point3d> > > objectPoints_all;
	vector<vector<Size> > boardsizes_all;

	Optimizer *optimizer = new Optimizer(n_boards, cameraMatrix);
	int *board_freq_count = new int[n_boards];
	memset(board_freq_count, 0, sizeof(int)*n_boards);
	for (int i_view = 0; i_view < n; i_view++)
	{
		CamView current_view;
		mxArray *cur_boards = mxGetCell(chessboards, (mwIndex)i_view);
		mxArray *cur_corners = mxGetCell(corners, (mwIndex)i_view);
		mxArray *point_coords = mxGetField(cur_corners, 0, "p");
		double *coords_table = mxGetPr(point_coords);
		int table_offset = mxGetM(point_coords);
		//cout << "offset is " << table_offset << endl;

		// decode chessboard 3-D subpixel precision coordinates
		int cur_n_board = mxGetN(cur_boards);
		current_view.board_index = new int[n_boards];
		for (int i_board = 0; i_board < n_boards; i_board++) current_view.board_index[i_board] = -1;
		vector<Point3d> camera_vecs;	
		vector<Mat> pic_extrinsics;

		vector<Mat> tvecs_pic;
		vector<Mat> rvecs_pic;
		vector<Size> boardsizes_pic;

		for (int i_board = 0; i_board < cur_n_board; i_board++)
		{
			vector<Point2d> imagePoints;
			vector<Point3d> objectPoints;

			int index = roundl(matching_table[i_view + i_board * n]) - 1;
			if (index >= 0)
			{
				current_view.board_index[index] = i_board;
				board_freq_count[index]++;
			}

			mxArray *board_i = mxGetCell(cur_boards, (mwIndex)i_board);
			int row = mxGetM(board_i);
			int col = mxGetN(board_i);
			double *board_ids = mxGetPr(board_i);
			//cout << "dimension is " << row << " x " << col << endl;
			//cout << "board " << i_board << " in picture " << i << " has " << row << " rows and " << col << " columns of inner corners" << endl;
			for (int r_id = 0; r_id < row; r_id++)
			{
				for (int c_id = 0; c_id < col; c_id++)
				{
					int index = r_id + c_id * row;	// this is special for Matlab convention
					int point_id = roundl(board_ids[index]) - 1;	// minus 1 due to 0-start in C++
					double x = coords_table[point_id];
					double y = coords_table[point_id + table_offset];
					imagePoints.push_back(Point2d(x, y));
				}
			}
			//int scaler = row > col ? row : col;	// chessboard_size corresponds to the lengthy edge of your chessboard
			//objectPoints = generateChessboard3d(row, col, chessboard_size / (scaler + 1));
			objectPoints = generateChessboard3d(row, col, chessboard_size);
			current_view.ObjectPoints.push_back(objectPoints);
			current_view.ImagePoints.push_back(imagePoints);

			boardsizes_pic.push_back(Size(row, col));
			
			// solve extrinsic camera parameters of this board in this picture
			Mat rvec(3, 1, CV_64F);
			Mat tvec(3, 1, CV_64F);			
			solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
			// compute the camera coordinate relative to this chessboard and push to stack
			Mat R;
			Rodrigues(rvec, R); // R is 3x3

			Mat RT, RT_full;
			hconcat(R, tvec, RT);
			Mat bottom = Mat::zeros(1, 4, CV_64F);
			bottom.at<double>(0, 3) = 1;
			vconcat(RT, bottom, RT_full);
			current_view.extrinsic_mat.push_back(RT_full);
			current_view.board_sizes.push_back(Size(row, col));
		}
		optimizer->camera_views.push_back(current_view);
		//if (i_view >= 8) break;
	}
	int max_freq = 0;
	for (int i = 0; i < n_boards; i++)
	{
		if (max_freq < board_freq_count[i])
		{
			max_freq = board_freq_count[i];
			optimizer->main_board = i;
		}
	}
	*/
#if SOLVECHESS
	optimizer->optimize(500);
	optimizer->visualize(imgpath);
	FileStorage fs("camerapos.yml", FileStorage::WRITE);
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		fs << optimizer->mat_label[i_board] << optimizer->relative_mat[i_board];
	}
	fs.release();
#else
	optimizer->initialize();
	FileStorage fs("camerapos.yml", FileStorage::READ);
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		fs[optimizer->mat_label[i_board]] >> optimizer->relative_mat[i_board];
	}
	fs.release();
#endif

#if SOLVERICOH	
	MapLocator *RicohLocator = new MapLocator(n_boards, 2688, true);
	sprintf(command, "load('ricoh.mat');");
	engEvalString(ep, command);

	mxArray *chessboard = engGetVariable(ep, "chessboard");
	mxArray *corner = engGetVariable(ep, "corner");
	mxArray *point_coords = mxGetField(corner, 0, "p");
	double *coord_table = mxGetPr(point_coords);
	int table_offset = mxGetM(point_coords);
	RicohLocator->board_index = new int[n_boards];
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		//RicohLocator->relative_mat.push_back(optimizer->relative_mat[i_board]);
		RicohLocator->relative_mat = optimizer->relative_mat;
		RicohLocator->board_index[i_board] = -1;
	}
	int cur_n_board = mxGetN(chessboard);
	for (int i_board = 0; i_board < cur_n_board; i_board++)
	{
		vector<Point2d> imagePoints;
		vector<Point3d> objectPoints;

		mxArray *board_i = mxGetCell(chessboard, (mwIndex)i_board);
		int row = mxGetM(board_i);
		int col = mxGetN(board_i);
		double *board_ids = mxGetPr(board_i);
		//cout << "dimension is " << row << " x " << col << endl;
		//cout << "board " << i_board << " in picture " << i << " has " << row << " rows and " << col << " columns of inner corners" << endl;

		bool was_detected = false;
		for (int j_board = 0; j_board < n_boards; j_board++)
		{
			Size ref_size = optimizer->camera_views[0].board_sizes[j_board];
			if ((ref_size.height == row && ref_size.width == col) || 
				(ref_size.height == col && ref_size.width == row))
			{
				RicohLocator->board_index[j_board] = i_board;
				was_detected = true;
				break;
			}
		}
		//if (!was_detected) continue;

		for (int r_id = 0; r_id < row; r_id++)
		{
			for (int c_id = 0; c_id < col; c_id++)
			{
				int index = r_id + c_id * row;	// this is special for Matlab convention
				int point_id = roundl(board_ids[index]) - 1;	// minus 1 due to 0-start in C++
				double x = coord_table[point_id];
				double y = coord_table[point_id + table_offset];
				imagePoints.push_back(Point2d(x, y));
			}
		}
		//int scaler = row > col ? row : col;	// chessboard_size corresponds to the lengthy edge of your chessboard
		//objectPoints = generateChessboard3d(row, col, chessboard_size / (scaler + 1));
		objectPoints = generateChessboard3d(row, col, chessboard_size);
		RicohLocator->ObjectPoints.push_back(objectPoints);
		RicohLocator->ImagePoints.push_back(imagePoints);
		RicohLocator->board_sizes.push_back(Size(row, col));
	}
	if (RicohLocator->ObjectPoints.empty())
	{
		cout << "no valid chessboard in the environment map!" << endl;
		exit(1);
	}
	RicohLocator->optimize(1000);
#endif



	// Now we can start calculating transformations and errors
	/*	
	ofstream oFile;
	oFile.open("PrecisionTestResultsForMultiple.csv", ios::app | ios::out);

	oFile << matname << endl;
	oFile << "source,";
	FILE *fin;
	fin = fopen(imgpath, "r");
	vector<vector<double> > reprojErrors_all;
	int n_views = optimizer->camera_views.size();
	for (int i_view = 0; i_view < n_views; i_view++)
	{
		char picname[MAX_LEN];
		sprintf(picname, "picture %d", i_view);
		oFile << picname << ",";
		fscanf(fin, "%s", &picname);
		cout << "reading image " << picname << endl;
		Mat view = imread(picname);
		vector<double> reprojErrors_pic;
		int n_boards = boardsizes_all[i].size();
		for (int j = 0; j < n_boards; j++)
		{
			{
				Mat reprojMat, reprojTvec, reprojRvec;
				gemm(solutions_all[i], camera_extrinsics[i + 1][j], 1, NULL, 0, reprojMat);
				Rodrigues(reprojMat(Rect(0, 0, 3, 3)), reprojRvec);
				reprojTvec = reprojMat(Rect(3, 0, 1, 3));
				reprojError = computeReprojectionErrors(objectPoints_all[i + 1][j], imagePoints_all[i][j],
					reprojRvec, reprojTvec, cameraMatrix, distCoeffs, reprojectedPoints);
			}
			reprojErrors_pic.push_back(reprojError);
			drawChessboardCorners(view, boardsizes_all[i][j], reprojectedPoints, true);
		}
		sprintf(picname, "optimized%d.png", i);
		imwrite(picname, view);
		reprojErrors_all.push_back(reprojErrors_pic);
	}
	fclose(fin);
	fs.release();
	*/

	return 0;
}