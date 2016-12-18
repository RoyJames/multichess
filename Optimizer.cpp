#include "Optimizer.h"



Optimizer::Optimizer()
{
}

Optimizer::Optimizer(int _n_boards, Mat _intrinsic_mat, Mat _distortion_mat):
	n_boards(_n_boards),intrinsic_mat(_intrinsic_mat), distortion_mat(_distortion_mat)
{
}

Optimizer::~Optimizer()
{
}

void Optimizer::initialize()
{
	// initialize relative transform matrices
	int *board_freq = new int[n_boards];
	int max_freq = 0;
	main_board = -1;
	memset(board_freq, 0, sizeof(board_freq));

	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		// initialize relative matrices
		relative_mat.push_back(Mat::eye(4, 4, CV_64F));
		for (int i_view = 0; i_view < camera_views.size(); i_view++)
		{
			// if this board is FULLY detected under this view then use it
			if (camera_views[i_view].ImagePoints[i_board].size() > 0 &&
				camera_views[i_view].ImagePoints[i_board].size() ==
				camera_views[i_view].ObjectPoints[i_board].size())
			{
				board_freq[i_board]++;
				// solve extrinsic camera parameters of this board in this picture
				Mat rvec(3, 1, CV_64F);
				Mat tvec(3, 1, CV_64F);

				solvePnP(camera_views[i_view].ObjectPoints[i_board], 
					camera_views[i_view].ImagePoints[i_board], 
					intrinsic_mat, distortion_mat, rvec, tvec);
				// compute the camera coordinate relative to this chessboard and push to stack
				Mat R;
				Rodrigues(rvec, R); // R is 3x3
				Mat RT, RT_full;
				hconcat(R, tvec, RT);
				Mat bottom = Mat::zeros(1, 4, CV_64F);
				bottom.at<double>(0, 3) = 1;
				vconcat(RT, bottom, RT_full);
				camera_views[i_view].extrinsic_mat.push_back(RT_full);
			}
			else
			{	// assign it an empty Matrix as position holder
				camera_views[i_view].extrinsic_mat.push_back(Mat::zeros(1, 4, CV_64F));
			}
		}
		if (max_freq < board_freq[i_board])
		{
			max_freq = board_freq[i_board];
			main_board = i_board;
		}
	}

	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		if (i_board == main_board) continue;
		vector<Mat> A_buf, B_buf;
		for (int i_view = 0; i_view < camera_views.size(); i_view++)
		{
			// check whether the main board can be detected in this view
			if (camera_views[i_view].ImagePoints[main_board].size() <= 0) continue;
			camera_views[i_view].main_extrinsic_mat =
				camera_views[i_view].extrinsic_mat[main_board];
			// see if this board can be detected in this view
			if (camera_views[i_view].ImagePoints[i_board].size() <= 0 ||
				camera_views[i_view].ImagePoints[i_board].size() !=
				camera_views[i_view].ObjectPoints[i_board].size()) continue;
			A_buf.push_back(camera_views[i_view].main_extrinsic_mat);
			B_buf.push_back(camera_views[i_view].extrinsic_mat[i_board]);
		}
		Mat A, B, x;
		vconcat(A_buf, A);
		vconcat(B_buf, B);
		// use least square to intialize relative matrices
		solve(A, B, x, DECOMP_SVD);
		relative_mat[i_board] = x;
	}	
	
	// for those frames in which the main board is invisible, recover one
	// in fact, we are not using the estimation currently
	for (int i_view = 0; i_view < camera_views.size(); i_view++)
	{
		if (camera_views[i_view].ImagePoints[main_board].size() > 0) continue;
		// if main board in this view is invisible
		Mat estimator = Mat::zeros(4, 4, CV_64F);
		int cnt = 0;
		for (int i_board = 0; i_board < n_boards; i_board++)
		{
			if (camera_views[i_view].ImagePoints[i_board].size() <= 0 
				|| i_board == main_board) continue;
			gemm(camera_views[i_view].extrinsic_mat[i_board], relative_mat[i_board].inv(),
				1.0, estimator, 1.0, estimator);
			cnt++;
		}
		if (cnt > 0) camera_views[i_view].main_extrinsic_mat = estimator * (1.0 / cnt);
	}
}

double Optimizer::get_temperature()
{
	//return (double)(total_iter - current_iter) / total_iter;
	return 0.1 / (current_iter + 1);
	//return 0.01;
}

void Optimizer::optimize(int max_iter)
{
	cout << "start initialization..." << endl;
	initialize();
	cout << "finish initialization..." << endl;
	visualize("pic_list.txt");
	total_iter = max_iter;
	for (current_iter = 0; current_iter < max_iter; current_iter++)
	{
		cout << "iteration " << current_iter + 1 << endl;
		//for (int sub_iter = 0; sub_iter < ALTERNATE_INTERVAL; sub_iter++)
		//{
		//	update_main();
		//}
		//closeup();
		//visualize("list.txt");
		//waitKey(0);
		update_relative();
		if (current_iter % 100 == 0) {
			closeup();
			visualize("pic_list.txt");
		}
		//waitKey(0);
	}
	closeup();
}

void Optimizer::closeup()
{
	int n_view = camera_views.size();
	for (int i_view = 0; i_view < n_view; i_view++)
	{
		for (int i_board = 0; i_board < n_boards; i_board++)
		{
			if (camera_views[i_view].ImagePoints[i_board].size() <= 0 ||
				camera_views[i_view].ImagePoints[i_board].size() !=
				camera_views[i_view].ObjectPoints[i_board].size() ||
				camera_views[i_view].ImagePoints[main_board].size() <= 0) continue;
			camera_views[i_view].extrinsic_mat[i_board] = 
				camera_views[i_view].main_extrinsic_mat * relative_mat[i_board];
		}
	}
}

void Optimizer::visualize(char *imgpath)
{
	FILE *fin;
	fin = fopen(imgpath, "r");
	int n_views = -1;
	fscanf(fin, "%d", &n_views);
	char picname[MAX_LEN];
	FileStorage fs_extrinsic("extrinsics.yml", FileStorage::WRITE);
	for (int i_view = 0; i_view < n_views; i_view++)
	{		
		fscanf(fin, "%s", &picname); 
		Mat view = imread(picname);
		vector<double> reprojErrors_pic;
		for (int i_board = 0; i_board < n_boards; i_board++)
		{
			if (camera_views[i_view].ImagePoints[i_board].size() <= 0 ||
				camera_views[i_view].ImagePoints[i_board].size() !=
				camera_views[i_view].ObjectPoints[i_board].size())	continue;
			Mat cur_extrinsic = camera_views[i_view].extrinsic_mat[i_board];

			stringstream cur_label;
			cur_label << "view" << i_view << "ext" << i_board;
			fs_extrinsic << cur_label.str() << cur_extrinsic;

			Mat reprojTvec, reprojRvec;
			Rodrigues(cur_extrinsic(Rect(0, 0, 3, 3)), reprojRvec);
			reprojTvec = cur_extrinsic(Rect(3, 0, 1, 3));
			vector<Point2d> reprojectedPoints; 
			double reprojError = computeReprojectionErrors(camera_views[i_view].ObjectPoints[i_board],
				camera_views[i_view].ImagePoints[i_board], reprojRvec, reprojTvec, intrinsic_mat, 
				distortion_mat, reprojectedPoints);
			reprojErrors_pic.push_back(reprojError);
			//cout << "reprojected points:" << reprojectedPoints.size() <<
			//	" board size:" << camera_views[i_view].board_sizes[cur_board_index] << endl;
			vector<Point2f> drawnPoints;
			Mat(reprojectedPoints).convertTo(drawnPoints, Mat(drawnPoints).type());
			drawChessboardCorners(view, board_sizes[i_board], drawnPoints, true);

			drawAxis(view, reprojRvec, reprojTvec, intrinsic_mat, distortion_mat);
		}
		/*
		for (int i_sample = 0; i_sample < camera_views[i_view].SampledPoints.size(); i_sample++)
		{
			circle(view, camera_views[i_view].SampledPoints[i_sample], 10, Scalar(0, 0, 255), 2);
		}*/
		Mat small_view;
		resize(view, small_view, Size(0, 0), 0.5, 0.5);
		//imshow(picname, small_view);
		sprintf(picname, "iterations/pic%diter%d.png", i_view, current_iter);
		if (WRITE_IMAGE) {
			if (current_iter != total_iter)
			{
				imwrite(picname, small_view);
			}
			else {
				imwrite(picname, view);
			}
		}		
		//cout << "reprojection error of view " << i_view << ":" << endl;
		for (int iter = 0; iter < reprojErrors_pic.size(); iter++)
		{
			cout << reprojErrors_pic[iter] << " ";
		}
		cout << endl;
	}
	fclose(fin);
	fs_extrinsic.release();
}

double Optimizer::computeReprojectionErrors(const vector<Point3d>& objectPoints,
	const vector<Point2d>& imagePoints,
	const Mat& rvec, const Mat& tvec,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<Point2d>& imagePoints2)
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

void Optimizer::drawAxis(Mat &img, const Mat& rvec, const Mat& tvec,
	const Mat& cameraMatrix, const Mat& distCoeffs)
{
	int axis_len = 4;
	vector<Point3d> axis_control;
	Point3d origin(0, 0, 0);
	Point3d x_end(CHESSBOARD_SIZE * axis_len, 0, 0);
	Point3d y_end(0, CHESSBOARD_SIZE * axis_len, 0);
	Point3d z_end(0, 0, CHESSBOARD_SIZE * axis_len);
	axis_control.push_back(origin);
	axis_control.push_back(x_end);
	axis_control.push_back(y_end);
	axis_control.push_back(z_end);

	vector<Point2d> axis_proj;
	projectPoints(Mat(axis_control), rvec, tvec, cameraMatrix, distCoeffs, axis_proj);
	line(img, axis_proj[0], axis_proj[1], Scalar(0, 0, 255), 5);	// draw x axis
	line(img, axis_proj[0], axis_proj[2], Scalar(0, 255, 0), 5);	// draw y axis
	line(img, axis_proj[0], axis_proj[3], Scalar(255, 0, 0), 5);	// draw z axis
}

Mat Optimizer::solve_increment(const Mat &Jr, const Mat &r)
{
	assert(Jr.rows == r.rows);
	Mat A, B, x;
	gemm(Jr, Jr, 1.0, NULL, 0, A, GEMM_1_T);
	gemm(Jr, r, -1.0, NULL, 0, B, GEMM_1_T);
	solve(A, B, x, DECOMP_SVD);
	double temperature = get_temperature();
	if(norm(x) > temperature) x *= temperature / norm(x);
	//x *= 0.1;
	//cout << "incremental: ";
	//printMat(x.t());
	cout<< " norm:" << norm(x) << " temperature:" << temperature << endl;
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
	//cout << "code 1 debugger" << endl;
	//printMat(incremental);
	//orthogonalize_transform(incremental);
	return incremental.clone();
}

Mat Optimizer::compute_Jacobian_main(const Point3d &p, const Point2d &M, 
	const Mat &main_mat, const Mat &trans_mat, double &residual)
{
	Mat p_vec = Mat(p, CV_64F);
	//cout << "object:" << p << " image:" << M << endl;
	/*	
	cout << "main matrix:" << endl;
	printMat(main_mat);
	cout << "transform matrix:" << endl;
	printMat(trans_mat);
	*/
	// we compute the Jacobian (gradient) for the residual by chain rule
	// 1. Jacobian of g()
	Mat Tp = main_mat(Rect(0, 0, 3, 3)) * p_vec + main_mat(Rect(3, 0, 1, 3));
	double x = Tp.at<double>(0);
	double y = Tp.at<double>(1);
	double z = Tp.at<double>(2);
	double J_g_buffer[9] = {
		y*trans_mat.at<double>(0,2) - z*trans_mat.at<double>(0,1), 
		z*trans_mat.at<double>(0,0) - x*trans_mat.at<double>(0,2),
		x*trans_mat.at<double>(0,1) - y*trans_mat.at<double>(0,0),
		y*trans_mat.at<double>(1,2) - z*trans_mat.at<double>(1,1), 
		z*trans_mat.at<double>(1,0) - x*trans_mat.at<double>(1,2),
		x*trans_mat.at<double>(1,1) - y*trans_mat.at<double>(1,0),
		y*trans_mat.at<double>(2,2) - z*trans_mat.at<double>(2,1), 
		z*trans_mat.at<double>(2,0) - x*trans_mat.at<double>(2,2),
		x*trans_mat.at<double>(2,1) - y*trans_mat.at<double>(2,0)
	};
	Mat J_g = Mat(3, 3, CV_64F, J_g_buffer);
	hconcat(J_g, trans_mat(Rect(0, 0, 3, 3)), J_g);
	assert(J_g.rows == 3 && J_g.cols == 6);
	// 2. Jacobian of u()
	Mat cur_extrinsic_mat = trans_mat * main_mat;
	Mat proj = cur_extrinsic_mat(Rect(0, 0, 3, 3)) * p_vec + cur_extrinsic_mat(Rect(3, 0, 1, 3));
	double px = proj.at<double>(0);
	double py = proj.at<double>(1);
	double pz = proj.at<double>(2);
	double fx = intrinsic_mat.at<double>(0, 0);
	double fy = intrinsic_mat.at<double>(1, 1);
	double cx = intrinsic_mat.at<double>(0, 2);
	double cy = intrinsic_mat.at<double>(1, 2);
	double J_u_buffer[6] = {
		fx / pz, 0, -fx * px / (pz * pz),
		0, fy / pz, -fy * py / (pz * pz)
	};
	Mat J_u = Mat(2, 3, CV_64F, J_u_buffer);
	double u = fx * px / pz + cx;
	double v = fy * py / pz + cy;
	// 3. Jacobian of r()
	Mat J_r = Mat(1, 2, CV_64F);
	residual = norm(M - Point2d(u, v));
	J_r.at<double>(0) = -(M.x - u) / residual;
	J_r.at<double>(1) = -(M.y - v) / residual;
	// 4. gather chains
	Mat J_all = J_r * J_u * J_g;
	//gemm(J_u, J_g, 1.0, NULL, 0, J_all);
	//gemm(J_r, J_all, 1.0, NULL, 0, J_all);
	assert(J_all.rows == 1 && J_all.cols == 6);
	//cout << "reprojected (" << u << "," << v << ")" << endl;
	//__halt();
	return J_all.clone();
}

void Optimizer::update_main()
{
	int n_view = camera_views.size();
	double tot_err = 0;
	// the main extrinsic matrix for each view is independent of each other, optimize them separately
	for (int i_view = 0; i_view < n_view; i_view++)
	{
		CamView &current_view = camera_views[i_view];
		vector<Mat> Jacobi_rows;
		vector<double> residuals;
		for (int i_board = 0; i_board < n_boards; i_board++)
		{
			if (camera_views[i_view].ImagePoints[i_board].size() <= 0 ||
				camera_views[i_view].ImagePoints[i_board].size() !=
				camera_views[i_view].ObjectPoints[i_board].size()) continue;	// if some board is not detected in this view then skip it
			for (int i_point = 0; i_point < current_view.ImagePoints[i_board].size(); i_point++)
			{
				/*int w = board_sizes[index].width;
				int h = board_sizes[index].height;
				if (!(i_point == 0 ||
					i_point < 10 ||
					i_point == h - 1 ||
					i_point == (w - 1)*h ||
					i_point == w*h - 1)) continue;*/
				double residual;
				Jacobi_rows.push_back(compute_Jacobian_main(current_view.ObjectPoints[i_board][i_point],
					current_view.ImagePoints[i_board][i_point], current_view.main_extrinsic_mat, 
					relative_mat[i_board], residual));
				residuals.push_back(residual);
			}
		}
		Mat Jacobi_full;
		vconcat(Jacobi_rows, Jacobi_full);	// gather Jacobian matrix for this view
		Mat eta_mat = solve_increment(Jacobi_full, Mat(residuals));	
		//cout << "Jacobian norm=" << norm(Jacobi_full) << endl;
	
		//cout << "code 2 solved increment:" << endl;
		//printMat(eta_mat);
		//cout << "before update:" << endl;
		//printMat(current_view.main_extrinsic_mat);
		gemm(eta_mat, current_view.main_extrinsic_mat, 1.0, NULL, 0, current_view.main_extrinsic_mat);	// update current main extrinsic matrix	
		//cout << "after update, before orthogonalize:" << endl;
		//printMat(current_view.main_extrinsic_mat);
		orthogonalize_transform(current_view.main_extrinsic_mat);
		//cout << "after orthogonalize:" << endl;
		//printMat(current_view.main_extrinsic_mat);
		//exit(1);

		tot_err += norm(residuals);
	}
	cout << "total energy optimizing main: " << tot_err << endl;
}

Mat Optimizer::compute_Jacobian_relative(const Point3d &p, const Point2d &M, const Mat &main_mat, const Mat &trans_mat, double &residual)
{
	Mat p_vec = Mat(p);
	/*cout << "object:" << p << " image:" << M << endl;	
	cout << "main matrix:" << endl;
	printMat(main_mat);
	cout << "transform matrix:" << endl;
	printMat(trans_mat);*/
	
	// we compute the Jacobian (gradient) for the residual by chain rule
	// 1. Jacobian of g()
	//Mat cur_extrinsic_mat = trans_mat * main_mat;
	Mat Tp;
	gemm(trans_mat(Rect(0, 0, 3, 3)), p_vec, 1.0, trans_mat(Rect(3, 0, 1, 3)), 1.0, Tp);
	double x = Tp.at<double>(0);
	double y = Tp.at<double>(1);
	double z = Tp.at<double>(2);
	double J_g_buffer[9] = {
		y*main_mat.at<double>(0,2) - z*main_mat.at<double>(0,1),
		z*main_mat.at<double>(0,0) - x*main_mat.at<double>(0,2),
		x*main_mat.at<double>(0,1) - y*main_mat.at<double>(0,0),
		y*main_mat.at<double>(1,2) - z*main_mat.at<double>(1,1),
		z*main_mat.at<double>(1,0) - x*main_mat.at<double>(1,2),
		x*main_mat.at<double>(1,1) - y*main_mat.at<double>(1,0),
		y*main_mat.at<double>(2,2) - z*main_mat.at<double>(2,1),
		z*main_mat.at<double>(2,0) - x*main_mat.at<double>(2,2),
		x*main_mat.at<double>(2,1) - y*main_mat.at<double>(2,0)
	};
	Mat J_g = Mat(3, 3, CV_64F, J_g_buffer);
	hconcat(J_g, main_mat(Rect(0, 0, 3, 3)), J_g);
	assert(J_g.rows == 3 && J_g.cols == 6);

	Mat cur_extrinsic_mat;
	cur_extrinsic_mat = main_mat * trans_mat;
	gemm(cur_extrinsic_mat(Rect(0, 0, 3, 3)), p_vec, 1.0, cur_extrinsic_mat(Rect(3, 0, 1, 3)), 1.0, Tp);
	x = Tp.at<double>(0);
	y = Tp.at<double>(1);
	z = Tp.at<double>(2);

	// 2. Jacobian of u()
	double fx = intrinsic_mat.at<double>(0, 0);
	double fy = intrinsic_mat.at<double>(1, 1);
	double cx = intrinsic_mat.at<double>(0, 2);
	double cy = intrinsic_mat.at<double>(1, 2);
	double J_u_buffer[6] = {
		fx / z, 0, -fx*x / (z * z),
		0, fy / z, -fy*y / (z * z)
	};
	Mat J_u = Mat(2, 3, CV_64F, J_u_buffer);
	double u = fx*x / z + cx;
	double v = fy*y / z + cy;
	// 3. Jacobian of r()
	Mat J_r = Mat(1, 2, CV_64F);
	residual = norm(M - Point2d(u, v));
	J_r.at<double>(0) = -(M.x - u) / residual;
	J_r.at<double>(1) = -(M.y - v) / residual;
	// 4. gather chains
	Mat J_all = J_r * J_u * J_g;
	//gemm(J_u, J_g, 1.0, NULL, 0, J_all);
	//gemm(J_r, J_all, 1.0, NULL, 0, J_all);
	assert(J_all.rows == 1 && J_all.cols == 6);
	/*
	cout << "jacobians:" << endl;
	printMat(J_g);
	printMat(J_u);
	printMat(J_r);
	printMat(J_all);
	exit(1);
	*/
	return J_all.clone();
}

void Optimizer::update_relative()
{
	int n_view = camera_views.size();
	double tot_err = 0;
	// the relative matrix for the same chessboard is the same throughout all views
	// and each chessboard is also independet, so optimize them by gathering chessboard points from all views
	for (int i_board = 0; i_board < n_boards; i_board++)
	{
		if (i_board == main_board) continue;	// main board has identity relative matrix, no optimization needed
		vector<Mat> Jacobi_rows;
		vector<double> residuals;
		for (int i_view = 0; i_view < n_view; i_view++)
		{
			CamView &current_view = camera_views[i_view];
			if (camera_views[i_view].ImagePoints[i_board].size() <= 0 ||
				camera_views[i_view].ImagePoints[i_board].size() !=
				camera_views[i_view].ObjectPoints[i_board].size()) continue;	// if this board is not detected in this view then skip it
			for (int i_point = 0; i_point < current_view.ImagePoints[i_board].size(); i_point++)
			{
				/*int w = board_sizes[index].width;
				int h = board_sizes[index].height;
				if (!(i_point == 0 ||
					i_point < 0 ||
					i_point == h - 1 ||
					i_point == (w - 1)*h ||
					i_point == w*h - 1)) continue;
				current_view.SampledPoints.push_back(current_view.ImagePoints[index][i_point]);*/
				double residual;
				Jacobi_rows.push_back(compute_Jacobian_relative(current_view.ObjectPoints[i_board][i_point],
					current_view.ImagePoints[i_board][i_point], current_view.main_extrinsic_mat,
					relative_mat[i_board], residual));
				residuals.push_back(residual);
			}
		}
		Mat Jacobi_full;
		vconcat(Jacobi_rows, Jacobi_full);	// gather Jacobian matrix for this view
		Mat eta_mat = solve_increment(Jacobi_full, Mat(residuals));
		cout << "Jacobian norm=" << norm(Jacobi_full) << endl;

		gemm(eta_mat, relative_mat[i_board], 1.0, NULL, 0, relative_mat[i_board]);	// update current main extrinsic matrix
		orthogonalize_transform(relative_mat[i_board]);
		tot_err += norm(residuals);
	}
	cout << "total reprojection error optimizing relatives: " << tot_err << endl;
}

void Optimizer::printMat(const Mat &target)
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

void Optimizer::orthogonalize_transform(Mat &mat)
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