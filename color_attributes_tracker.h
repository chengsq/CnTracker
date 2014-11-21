/*
 * color_attributes_tracker.h
 *
 *  Created on: Nov 18, 2014
 *      Author: shiqingcheng
 */

#ifndef COLOR_ATTRIBUTES_TRACKER_H_
#define COLOR_ATTRIBUTES_TRACKER_H_

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
enum FEATURES {
	gray, cn
};
using namespace cv;

class ColorAttributesTracker {

public:
	ColorAttributesTracker();
	ColorAttributesTracker(Mat& rgb_mat, int x, int y,int width,int height, int id);
	void UpdateTracker(Mat& rgb_mat);
	virtual ~ColorAttributesTracker();
	void Update(Mat img);
	bool match_status_;
  Point pos;
  Size target_sz_;
  int id_;

private:
	vector<Mat> GetFeatureMap(Mat im_patch, FEATURES features, Mat w2c);
	void IM2C(vector<Mat>&outs, Mat im_patch, Mat w2c, int color);
	void GetSubwindow(const Mat im, Point pos, Size sz, const Mat non_pca_features, const Mat pca_features, const Mat w2c, Mat* out_npca, Mat* out_pca);
	void FeatureProjection(const Mat x_npca, const Mat x_pca, const Mat projection_matrix, const Mat cos_window, Mat& z);
	Mat DenseGaussKernel(const double sigma, const Mat x, const Mat y);
	void TrackerInit(Mat img);
	Mat LoadW2C(char* file_name);
	void Ndgrid(int x,int y,int size,Mat& out);
	void DimensionReduction();
	void DimensionReductionInit();
	bool dr_flag;

	float padding;
	float output_sigma_factor;
	float sigma;
	float lambda;
	float learning_rate;
	float compression_learning_rate;
	Mat non_compressed_features;
	Mat compressed_features;
	float output_sigma;
	int num_compressed_dim;
	Point init_pos_;
	Size sz;
	Mat cos_window;

	Mat xo_npca;
	Mat xo_pca;
	Mat z_npca;
	Mat z_pca;
	Mat w2c;
	Mat zp;
	Mat projection_matrix;

	Mat yf;
	Mat new_alphaf_num;
	Mat new_alphaf_den;

	Mat projection_variances;
	//vector<Mat> z_feature;
	Mat old_cov_matrix;
	Mat alphaf_num;
	Mat alphaf_den;
	PCA pca_;

};

#endif /* COLOR_ATTRIBUTES_TRACKER_H_ */