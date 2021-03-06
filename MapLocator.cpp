#include "MapLocator.h"



MapLocator::MapLocator()
{
}

MapLocator::MapLocator(int _n_boards, int _LL_height, bool _is_log_on):n_boards(_n_boards), LL_height(_LL_height), is_log_on(_is_log_on)
{

}
MapLocator::~MapLocator()
{
}

void MapLocator::drawAxis(Mat &img, int board_id)
{
	int axis_len = 4;
	int w = board_sizes[board_id].width;
	int h = board_sizes[board_id].height;
	Point2d origin = ImagePoints[board_id][0];
	Point2d x_end = ImagePoints[board_id][axis_len];
	Point2d y_end = ImagePoints[board_id][axis_len * w];

	line(img, origin, x_end, Scalar(0, 0, 255), 5);	// draw x axis
	line(img, origin, y_end, Scalar(0, 255, 0), 5);	// draw y axis
}

void MapLocator::mannual()
{
	initialize();
	namedWindow("LL View");
	Mat view = imread("chessroom.jpg");
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		if (ImagePoints[i_board].empty() ||
			ImagePoints[i_board].size() != ObjectPoints[i_board].size()) continue;
		vector<Point2f> drawnPoints;
		Mat(ImagePoints[i_board]).convertTo(drawnPoints, Mat(drawnPoints).type());
		drawChessboardCorners(view, board_sizes[i_board], drawnPoints, true);
	}

	double d0 = 0.01;
	double d1 = 0.01;
	while (true)
	{
		Mat updated_view = view.clone();
		vector<Point2f> drawnPoints;
		for (int i_board = 0; i_board < n_boards; i_board++)
		{
			Mat(ReprojImagePoints[i_board]).convertTo(drawnPoints, Mat(drawnPoints).type());
			drawChessboardCorners(updated_view, board_sizes[i_board], drawnPoints, true);
		}
		Mat small_view;
		resize(updated_view, small_view, Size(0, 0), 0.25, 0.25);
		imshow("LL View", small_view);
		double x, y, z, a, b, c;
		x = y = z = a = b = c = 0;
		while (true)
		{
			int key = waitKey();
			cout << key;
			if (key == '\r') break;
			switch (key) 
			{
			case 'w':b += d0; break;
			case 's':b -= d0; break;
			case 'a':a += d0; break;
			case 'd':a -= d0; break;
			case 'z':c += d0; break;
			case 'c':c -= d0; break;
			case '8':x += d1; break;
			case '2':x -= d1; break;
			case '4':y += d1; break;
			case '6':y -= d1; break;
			case '1':z += d1; break;
			case '3':z -= d1; break;
			default:
				break;
			}
		}
		cout << "eta updated" << endl;
		double incremental_buffer[16] = {
			cos(y)*cos(z),						cos(y)*sin(z),						-sin(y),	a,
			sin(x)*sin(y)*cos(z)-cos(x)*sin(z), sin(x)*sin(y)*sin(z)+cos(x)*cos(z), sin(x)*cos(y), b,
			cos(x)*sin(y)*cos(z)+sin(x)*sin(z), cos(x)*sin(y)*sin(z)-sin(x)*cos(z), cos(x)*cos(y), c,
			0, 0, 0, 1
		};
		Mat eta_mat(4, 4, CV_64F, incremental_buffer);
		gemm(eta_mat, main2sphere, 1.0, NULL, 0, main2sphere);
		orthogonalize_transform(main2sphere);

		double tot_err = 0;
		int tot_n = 0;
		for (int i_board = 0; i_board < n_boards; i_board++)
		{
			vector<Mat> Jacobi_rows;
			vector<double> residuals;
			if (ImagePoints[i_board].empty() ||
				ImagePoints[i_board].size() != ObjectPoints[i_board].size()) continue;	// if some board is not detected in this view then skip it
			for (int i_point = 0; i_point < ImagePoints[i_board].size(); i_point++)
			{
				double residual;
				Jacobi_rows.push_back(compute_Jacobian(ObjectPoints[i_board][i_point],
					ImagePoints[i_board][i_point], ReprojImagePoints[i_board][i_point], residual));
				residuals.push_back(residual);
			}
			tot_err += norm(residuals);
			tot_n += residuals.size();
		}
		cout << "total reprojection error: " << tot_err / sqrt(tot_n) << endl;
	}
}

void MapLocator::initialize()
{
	// what is a good initial guess?
	main2sphere = Mat::eye(4, 4, CV_64F);
	main2sphere.at<double>(0, 3) = -0.1;

	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		if (ImagePoints[i_board].empty() || 
			ImagePoints[i_board].size()!= ObjectPoints[i_board].size()) continue;
		for (int i_point = 0; i_point < ObjectPoints[i_board].size(); i_point++)
		{
			//cout << "board " << i_board << ",point " << i_point << ":" << ObjectPoints[cur_index][i_point] << endl;
			Mat translated = relative_mat[i_board](Rect(0, 0, 3, 3))*Mat(ObjectPoints[i_board][i_point],CV_64F) +
				relative_mat[i_board](Rect(3, 0, 1, 3));
			ObjectPoints[i_board][i_point].x = translated.at<double>(0);
			ObjectPoints[i_board][i_point].y = translated.at<double>(1);
			ObjectPoints[i_board][i_point].z = translated.at<double>(2);
			//cout << "board " << i_board << ",point " << i_point << ":" << ObjectPoints[cur_index][i_point] << endl;
		}
	}
	ReprojImagePoints = ImagePoints;
	if (is_log_on)
	{
		logFile.open("IterationLog.csv", ios::app | ios::out);
		logFile << "Iteration,Board ID,Jacobian Norm,Increment,Threshold,Residual Norm" << endl;
		logFile.close();
	}
}

double MapLocator::get_temperature()
{
	//return (double)(total_iter - current_iter) / total_iter;
	//if (current_iter > 100) return 1.0 / (current_iter*5 + 1);
	//return 1.0 / (sqrt(current_iter) + 1);
	if (current_iter < 100)
	{
		return 0.1;
	}
	else {
		return 10.0 / (current_iter + 1);
	}
}

void MapLocator::optimize(int max_iter)
{
	cout << "start initialization..." << endl;
	initialize();
	cout << "finish initialization..." << endl;
	closeup();
	total_iter = max_iter;
	for (current_iter = 0; current_iter < max_iter; current_iter++)
	{
		cout << "iteration " << current_iter + 1 << endl;
		update();
		if (current_iter % 50 == 0) {
			closeup();
			visualize("chessroom.jpg");
		}
		//exit(1);
	}
	closeup();
}

void MapLocator::closeup()
{
	if (is_log_on)
	{
		logFile.close();
	}
}

void MapLocator::visualize(char *imgpath)
{
	Mat view = imread(imgpath);
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		if (ImagePoints[i_board].empty() ||
			ImagePoints[i_board].size() != ObjectPoints[i_board].size()) continue;
		vector<Point2f> drawnPoints;
		Mat(ReprojImagePoints[i_board]).convertTo(drawnPoints, Mat(drawnPoints).type());
		drawChessboardCorners(view, board_sizes[i_board], drawnPoints, true);

		Mat(ImagePoints[i_board]).convertTo(drawnPoints, Mat(drawnPoints).type());
		drawChessboardCorners(view, board_sizes[i_board], drawnPoints, true);

		drawAxis(view, i_board);
	}
	stringstream cur_label;
	cur_label << "iterations/chessroom_iter" << current_iter << ".png";

	Mat small_view;
	resize(view, small_view, Size(0, 0), 0.5, 0.5);
	imwrite(cur_label.str(), small_view);
}


Mat MapLocator::solve_increment(const Mat &Jr, const Mat &r, Mat &x)
{
	assert(Jr.rows == r.rows);
	Mat A, B;
	gemm(Jr, Jr, 1.0, NULL, 0, A, GEMM_1_T);
	gemm(Jr, r, -1.0, NULL, 0, B, GEMM_1_T);
	solve(A, B, x, DECOMP_SVD);

	double temperature = get_temperature();
	if (norm(x) > temperature) x *= temperature / norm(x);
	//x *= 0.1;
	//cout << "incremental: ";
	//printMat(x.t());
	cout << " norm:" << norm(x) << " temperature:" << temperature << endl;
	double alpha = x.at<double>(0);
	double beta = x.at<double>(1);
	double gamma = x.at<double>(2);
	double a = x.at<double>(3);
	double b = x.at<double>(4);
	double c = x.at<double>(5);
	double incremental_buffer[16] = {
		1, -gamma, beta, a,
		gamma, 1, -alpha, b,
		-beta, alpha, 1, c,
		0, 0, 0, 1
	};
	Mat incremental(4, 4, CV_64F, incremental_buffer);
	//printMat(incremental);
	//orthogonalize_transform(incremental);
	return incremental.clone();
}


Mat MapLocator::compute_Jacobian(const Point3d &p, const Point2d &M, Point2d &projected, double &residual)
{
	// we compute the Jacobian (gradient) for the residual by chain rule
	// 1. Jacobian of g()
	Mat p_vec = Mat(p);
	Mat Tp;
	gemm(main2sphere(Rect(0, 0, 3, 3)), p_vec, 1.0, main2sphere(Rect(3, 0, 1, 3)), 1.0, Tp);
	double x = Tp.at<double>(0);
	double y = Tp.at<double>(1);
	double z = Tp.at<double>(2);
	double J_g_buffer[18] = {
		0,z,-y,1,0,0,
		-z,0,x,0,1,0,
		y,-x,0,0,0,1
	};
	Mat J_g = Mat(3, 6, CV_64F, J_g_buffer);


	// 2. Jacobian of T()
	double L = x*x + y*y + z*z;
	double L3_2 = sqrt(L*L*L);
	double L1_2 = sqrt(L);
	double J_T_buffer[9] = {
		1.0 / L1_2 - x*x / L3_2, -x*y / L3_2, -x*z / L3_2,
		-x*y / L3_2, 1.0 / L1_2 - y*y / L3_2, -y*z / L3_2,
		-x*z / L3_2, -y*z / L3_2, 1.0 / L1_2 - z*z / L3_2
	};
	Mat J_T = Mat(3, 3, CV_64F, J_T_buffer);
	x /= L1_2;
	y /= L1_2;
	z /= L1_2;
	
	// 3. Jacobian of u()
	double scaler = LL_height / PI;
	double J_u_buffer[6] = {
		-z / (x*x + z*z) * scaler, 0, x / (x*x + z*z) * scaler,
		0, -1.0 / sqrt(1 - y*y) * scaler, 0	// pay attention to the sign of dy
	};
	Mat J_u = Mat(2, 3, CV_64F, J_u_buffer);
	double u = (1 + atan2(x, -z) / PI)*LL_height;
	//double v = (1 - acos(y) / PI)*LL_height;	// pay attention to the sign of y
	double v = (acos(y) / PI)*LL_height;
	projected = Point2d(u, v);

	// 4. Jacobian of r()
	Mat J_r = Mat(1, 2, CV_64F);
	residual = norm(M - Point2d(u, v));
	J_r.at<double>(0) = -(M.x - u) / residual;
	J_r.at<double>(1) = -(M.y - v) / residual;
	// 5. gather chains
	Mat J_all = J_r * J_u * J_T * J_g;
	//gemm(J_u, J_g, 1.0, NULL, 0, J_all);
	//gemm(J_r, J_all, 1.0, NULL, 0, J_all);
	assert(J_all.rows == 1 && J_all.cols == 6);

	bool nanflag = false;
	for (int i = 0; i < 6; i++)
	{
		if (isnan(J_all.at<double>(i)))
		{
			nanflag = true;
			break;
		}
	}
	if (nanflag)
	{ 
	cout << "object:" << p << " image:" << M << endl;
	cout << "transform matrix:" << endl;
	printMat(main2sphere);
	cout << "jacobians:" << endl;
	printMat(J_g);
	printMat(J_T);
	printMat(J_u);
	printMat(J_r);
	printMat(J_all);
	exit(1);
	}
	
	return J_all.clone();
}

void MapLocator::update()
{
	double tot_err = 0;
	int tot_n = 0;
	// the relative matrix for the same chessboard is the same throughout all views
	// and each chessboard is also independet, so optimize them by gathering chessboard points from all views
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		vector<Mat> Jacobi_rows;
		vector<double> residuals;	
		if (ImagePoints[i_board].empty() ||
			ImagePoints[i_board].size() != ObjectPoints[i_board].size()) continue;	// if some board is not detected in this view then skip it
		for (int i_point = 0; i_point < ImagePoints[i_board].size(); i_point++)
		{
			int w = board_sizes[i_board].width;
			int h = board_sizes[i_board].height;
			double residual;
			Jacobi_rows.push_back(compute_Jacobian(ObjectPoints[i_board][i_point],
				ImagePoints[i_board][i_point], ReprojImagePoints[i_board][i_point], residual));
			residuals.push_back(residual);
		}
		Mat Jacobi_full, x;
		vconcat(Jacobi_rows, Jacobi_full);	// gather Jacobian matrix for this view
		Mat eta_mat = solve_increment(Jacobi_full, Mat(residuals), x);
		cout << "Jacobian norm=" << norm(Jacobi_full) << endl;
		

		gemm(eta_mat, main2sphere, 1.0, NULL, 0, main2sphere);	// update current target matrix
		orthogonalize_transform(main2sphere);
		tot_err += norm(residuals);
		tot_n += residuals.size();

		if (is_log_on)
		{ 
			logFile.open("IterationLog.csv", ios::app | ios::out);
			logFile << current_iter << "," << i_board << "," << norm(Jacobi_full) <<
				"," << norm(x) << "," << get_temperature() << "," << norm(residuals) << endl;
			logFile.close();
		}
	}
	cout << "total reprojection error: " << tot_err / sqrt(tot_n) << endl;
}

void MapLocator::printMat(const Mat &target)
{
	int row = target.rows;
	int col = target.cols;
	cout << "[";
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << target.at<double>(i, j) << "\t";
		}
		if (i < row - 1) cout << endl;
	}
	cout << "]" << endl << endl;
}

void MapLocator::orthogonalize_transform(Mat &mat)
{
	Point3d x(mat.at<double>(0, 0), mat.at<double>(0, 1), mat.at<double>(0, 2));
	Point3d y(mat.at<double>(1, 0), mat.at<double>(1, 1), mat.at<double>(1, 2));
	Point3d z(mat.at<double>(2, 0), mat.at<double>(2, 1), mat.at<double>(2, 2));

	x = x * (1.0 / norm(x));
	y = y * (1.0 / norm(y));
	z = z * (1.0 / norm(z));

	Point3d x0(0.f, 0.f, 0.f), y0(0.f, 0.f, 0.f), z0(0.f, 0.f, 0.f);
	Point3d x1, y1, z1;

	x0 += x;
	x1 = x;
	z1 = x.cross(y);
	z1 = z1 * (1.0 / norm(z1));
	y1 = z1.cross(x1);
	y1 = y1 * (1.0 / norm(y1));
	z0 += z1;
	y0 += y1;

	y0 += y;
	x1 = y.cross(z);
	x1 = x1 * (1.0 / norm(x1));
	z1 = x1.cross(y);
	z1 = z1 * (1.0 / norm(z1));
	x0 += x1;
	z0 += z1;

	z0 += z;
	y1 = z.cross(x);
	y1 = y1 * (1.0 / norm(y1));
	x1 = y1.cross(z);
	x1 = x1 * (1.0 / norm(x1));
	y0 += y1;
	x0 += x1;

	x0 = x0 * (1.0 / norm(x0));
	y0 = y0 * (1.0 / norm(y0));
	z0 = z0 * (1.0 / norm(z0));

	z0 = x0.cross(y0);
	z0 = z0 * (1.0 / norm(z0));
	y0 = z0.cross(x0);

	mat.at<double>(0, 0) = x0.x;
	mat.at<double>(0, 1) = x0.y;
	mat.at<double>(0, 2) = x0.z;

	mat.at<double>(1, 0) = y0.x;
	mat.at<double>(1, 1) = y0.y;
	mat.at<double>(1, 2) = y0.z;

	mat.at<double>(2, 0) = z0.x;
	mat.at<double>(2, 1) = z0.y;
	mat.at<double>(2, 2) = z0.z;
}