
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/calib3d/calib3d.hpp>
#include <engine.h>
#include <fstream>

using namespace cv;
using namespace std;

#define MAX_LEN 1000
const double chessboard_size = 0.01;	// check this parameter with your render scene
float resolution = 1024;

vector<Point3f> generateChessboard3d(int row, int col, double edge)
{
	vector<Point3f> points;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			points.push_back(Point3f((float)j * edge, (float)i * edge, 0.0f));
			//cout << (float)j * edge << " " << (float)i * edge << endl;
		}
	}
	return points;
}

double dist3d(Point3f A, Point3f B)
{
	return sqrt((A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y) + (A.z - B.z)*(A.z - B.z));
}

void help()
{
	cout << "Input format:"
		<< "[name of the mat file (including .mat)]"
		<< "[resolution of pictures]"
		<< endl;
}

double computeReprojectionErrors(const vector<Point3f>& objectPoints,
	const vector<Point2f>& imagePoints,
	const Mat& rvec, const Mat& tvec,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<Point2f>& imagePoints2)
{
	projectPoints(Mat(objectPoints), rvec, tvec, cameraMatrix,
		distCoeffs, imagePoints2);
	//cout << (Mat(imagePoints).size == Mat(imagePoints2).size) << " "
	//	<< Mat(imagePoints).type() << " " << Mat(imagePoints2).type() << endl;
	double err = norm(Mat(imagePoints), Mat(imagePoints2), CV_L2);
	//cout << "done reprojecting" << endl;
	int n = imagePoints.size();

	return std::sqrt(err*err / n);
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

int main(int argc, char *argv[])
{
	
	if (argc < 2)
	{
		help();
		return 1;
	}
	char *matname = argv[1];
	//resolution = atof(argv[2]);
	char *imgpath = argv[2];


	Engine *ep; 
	if (!(ep = engOpen(NULL))) 
	{
		cout << "Can't start Matlab engine!" << endl;
		return 1;
	}

	FileStorage fs("camerapos.yml", FileStorage::WRITE);

	engEvalString(ep, "clear");
	cout << "engine started!" << endl;
	engEvalString(ep, "addpath('D:\\codes\\libcbdetect')");
	char command[MAX_LEN];
	sprintf(command, "load('%s');", matname);
	engEvalString(ep, command);

	mxArray *chessboards = engGetVariable(ep, "chessboards");
	mxArray *corners = engGetVariable(ep, "corners");
	mxArray *matching = engGetVariable(ep, "matching");

	int n = mxGetN(chessboards);
	//cout << "Total picture count: " << n << endl;
	
	// we have only one camera with known intrinsics
	Mat cameraMatrix = Mat::eye(3, 3, CV_32F);
	cameraMatrix.at<float>(0, 0) = 22170.18826229795;//1.634296875 * resolution;
	cameraMatrix.at<float>(1, 1) = 22170.18826229795;//1.634296875 * resolution;
	cameraMatrix.at<float>(0, 2) = 2375.5;//(resolution - 1) / 2;
	cameraMatrix.at<float>(1, 2) = 1583.5;//(resolution - 1) / 2;

	Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
	distCoeffs.at<double>(0) = -0.37077668666130514;
	distCoeffs.at<double>(1) = 8.0453921321123243;
	distCoeffs.at<double>(4) = 0.11371622504317971;

	vector<vector<Point3f> > camera_buffer;
	vector<vector<Mat> > camera_extrinsics;

	vector<vector<Mat> > tvecs_all;
	vector<vector<Mat> > rvecs_all;
	vector<vector<vector<Point2f> > > imagePoints_all;
	vector<vector<vector<Point3f> > > objectPoints_all;
	vector<vector<Size> > boardsizes_all;

	for (int i = 0; i < n; i++)
	{
		mxArray *cur_boards = mxGetCell(chessboards, (mwIndex)i);
		mxArray *cur_corners = mxGetCell(corners, (mwIndex)i);
		mxArray *point_coords = mxGetField(cur_corners, 0, "p");
		double *coords_table = mxGetPr(point_coords);
		int table_offset = mxGetM(point_coords);
		//cout << "offset is " << table_offset << endl;

		// decode chessboard 3-D subpixel precision coordinates
		int n_board = mxGetN(cur_boards);
		vector<Point3f> camera_vecs;	
		vector<Mat> pic_extrinsics;

		vector<Mat> tvecs_pic;
		vector<Mat> rvecs_pic;
		vector<vector<Point2f> > imagePoints_pic;
		vector<vector<Point3f> > objectPoints_pic;
		vector<Size> boardsizes_pic;

		for (int i_board = 0; i_board < n_board; i_board++)
		{
			vector<Point2f> imagePoints;
			vector<Point3f> objectPoints;

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
					imagePoints.push_back(Point2f(x, y));
					//cout << point_id << " ";
					//cout << "(" << x << "," << y << ") ";
				}
				//cout << endl;
			}
			//int scaler = row > col ? row : col;	// chessboard_size corresponds to the lengthy edge of your chessboard
			//objectPoints = generateChessboard3d(row, col, chessboard_size / (scaler + 1));
			objectPoints = generateChessboard3d(row, col, chessboard_size);
			imagePoints_pic.push_back(imagePoints);
			objectPoints_pic.push_back(objectPoints);
			boardsizes_pic.push_back(Size(row, col));

			
			//cout << row << "x" << col << ": " << chessboard_size / (scaler + 1) << endl;

			// solve extrinsic camera parameters of this board in this picture
			Mat rvec(3, 1, CV_64F);
			Mat tvec(3, 1, CV_64F);			

			solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
			tvecs_pic.push_back(tvec);
			rvecs_pic.push_back(rvec);

			// compute the camera coordinate relative to this chessboard and push to stack
			Mat R;
			Rodrigues(rvec, R); // R is 3x3

			Mat RT, RT_full;
			hconcat(R, tvec, RT);
			Mat bottom = Mat::zeros(1, 4, CV_64F);
			bottom.at<double>(0, 3) = 1;
			vconcat(RT, bottom, RT_full);
			char varname[MAX_LEN];
			sprintf(varname, "camera%dboard%d", i, i_board);
			fs << varname << RT_full;	// RT_full is 4x4

			pic_extrinsics.push_back(RT_full);
			/*
			Mat camera_coord(3, 1, CV_32F);
			Mat dummy = Mat::zeros(3, 1, CV_32F);
			gemm(R, tvec, 1.0, dummy, 0, camera_coord, GEMM_1_T);	
			
			vector<double> temp = getVector(camera_coord);
			camera_vecs.push_back(Point3f(temp[0], temp[1], temp[2]));
			*/
			
			// if you want to see how well it projects back
			/*
			vector<Point2f> projectedPoints;
			projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
			for (int k = 0; k < projectedPoints.size(); ++k)
			{
				cout << "Image point: " << imagePoints[k] << " Projected to " << projectedPoints[k] << endl;
			}
			*/
		}
		//camera_buffer.push_back(camera_vecs);
		camera_extrinsics.push_back(pic_extrinsics);

		imagePoints_all.push_back(imagePoints_pic);
		objectPoints_all.push_back(objectPoints_pic);
		tvecs_all.push_back(tvecs_pic);
		rvecs_all.push_back(rvecs_pic);
		boardsizes_all.push_back(boardsizes_pic);
	}

	// solver start
	double *matching_table = mxGetPr(matching);
	double single_estimates[4][10];	// for debugging only
	vector <Mat> solutions_all;
	//oFile << "joint,";
	for (int i = 0; i < n - 1; i++)
	{
		vector<Mat> pre = camera_extrinsics[i];
		vector<Mat> cur = camera_extrinsics[i + 1];

		int board_cnt = cur.size();

		cout << "Between picture " << i << " and " << i + 1 << ":" << endl;
		char varname[MAX_LEN];
		sprintf(varname, "transmat%d-%d", i, i + 1);
		vector<Mat> A_buffer, B_buffer;
		for (int k = 0; k < board_cnt; k++)
		{
			int correspondence = roundl(matching_table[i + k * (n - 1)]) - 1;
			if (correspondence < 0) continue;
			cout << "board " << k << " corresponds with board " << correspondence << endl;

			Mat A, B, X;
			transpose(pre[correspondence], A);
			transpose(cur[k], B);
			A_buffer.push_back(A);
			B_buffer.push_back(B);

			solve(B, A, X);
			transpose(X, X);
			double dist = 0;
			for (int k = 0; k < 3; k++)
			{
				dist += X.at<double>(k, 3) * X.at<double>(k, 3);
			}
			dist = sqrt(dist);
			single_estimates[k][i] = 0.01 - dist;
		}
		Mat src1, src2, solution, reproj;
		vconcat(A_buffer, src2);
		vconcat(B_buffer, src1);
		solve(src1, src2, solution, DECOMP_SVD);
		gemm(src1, solution, 1.0, NULL, 0, reproj);
		cout << "LS error = " << norm(reproj, src2, CV_L2) << endl;
		transpose(solution, solution);
		printMat(solution);
		solutions_all.push_back(solution);
		fs << varname << solution;
		double dist = 0;
		for (int k = 0; k < 3; k++)
		{
			dist += solution.at<double>(k, 3) * solution.at<double>(k, 3);
		}
		dist = sqrt(dist);
		//oFile << 0.01 - dist << ",";
	}
	// solver end

	// Now we can start calculating transformations and errors
	ofstream oFile;
	oFile.open("PrecisionTestResultsForMultiple.csv", ios::app | ios::out);

	oFile << matname << endl;
	oFile << "source,";
	FILE *fin;
	fin = fopen(imgpath, "r");
	vector<vector<double> > reprojErrors_all;
	for (int i = 0; i < n; i++)
	{
		char picname[MAX_LEN];
		sprintf(picname, "picture %d", i);
		oFile << picname << ",";
		fscanf(fin, "%s", &picname);
		cout << "reading image " << picname << endl;
		Mat view = imread(picname);
		vector<double> reprojErrors_pic;
		int n_boards = boardsizes_all[i].size();
		for (int j = 0; j < n_boards; j++)
		{
			vector<Point2f> reprojectedPoints;
			double reprojError;
			if (i == n - 1 || (i < n - 1 && boardsizes_all[i][j] != boardsizes_all[i+1][j])) 
			{
				reprojError = computeReprojectionErrors(objectPoints_all[i][j], imagePoints_all[i][j],
					rvecs_all[i][j], tvecs_all[i][j], cameraMatrix, distCoeffs, reprojectedPoints);
			}
			else 
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
		sprintf(picname, "reproject%d.png", i);
		imwrite(picname, view);
		reprojErrors_all.push_back(reprojErrors_pic);
	}
	fclose(fin);
	oFile << endl;
	int n_boards = boardsizes_all[0].size();
	for (int i = 0; i < n_boards; i++)
	{
		oFile << "board" << i << ",";
		for (int j = 0; j < n; j++)
		{
			oFile << reprojErrors_all[j][i] << ",";
		}
		oFile << endl;
	}

	/*
	for (int i = 1; i < n; i++)
	{
		char picname[MAX_LEN];
		sprintf(picname, "%d to %d", i - 1, i); 
		oFile << picname << ",";
	}
	oFile << endl;
	*/



	/*
	That solver was previously here.
	*/

	//oFile << endl;
	/*
	for (int i = 0; i < 4; i++)
	{
		oFile << "estimate " << i << ",";
		for (int j = 0; j < 10; j++)
		{
			oFile << single_estimates[i][j] << ",";
		}
		oFile << endl;
	}

	oFile << endl;*/
	fs.release();

	//engClose(ep);
	return 0;
}