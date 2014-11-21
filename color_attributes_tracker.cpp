/*
 * color_attributes_tracker.cpp
 *
 *  Created on: Nov 18, 2014
 *      Author: shiqingcheng
 */

#include "color_attributes_tracker.h"
#include <algorithm>
#include <iostream>
#include <stdio.h>

#define DEBUG_PRINT()  printf("%s %s %d \n",__FILE__,__FUNCTION__,__LINE__)
ColorAttributesTracker::ColorAttributesTracker() {

}
ColorAttributesTracker::~ColorAttributesTracker() {
}

ColorAttributesTracker::ColorAttributesTracker(Mat& rgb_mat, int x, int y,int width,int height,
                                               int id) {
#if 1
  id_ = id;
  padding = 1;
  output_sigma_factor = 0.0625;
  sigma = 0.2;
  lambda = 0.01;
  learning_rate = 0.075;
  compression_learning_rate = 0.15;
  non_compressed_features = (Mat_<int>(1, 1) << gray);
  //compressed_features = Mat();
  compressed_features = (Mat_<int>(1, 1) << cn);
  num_compressed_dim = 2;
  init_pos_.x = x;
  init_pos_.y = y;
  target_sz_.height = height;
  target_sz_.width = width;
  pos = init_pos_;
  sz.width = (1 + padding) * target_sz_.width;
  sz.height = (1 + padding) * target_sz_.height;
  match_status_ = false;

  float prod = target_sz_.height * target_sz_.width;
  output_sigma = sqrt(prod) * output_sigma_factor;

  Mat hann;
  createHanningWindow(hann, sz, CV_32F);  // TO-DO
  multiply(hann, hann, cos_window);
  Mat my = Mat(sz.height, sz.width, CV_32F);
  for (int j = 0; j < sz.height; ++j)
    for (int i = 0; i < sz.width; ++i) {
      int ai = i - sz.width / 2;
      int aj = j - sz.height / 2;
      float sqrt = ai * ai + aj * aj;
      my.at<float>(j, i) = exp(-0.5 * sqrt / output_sigma);
    }

  dft(my, yf, DFT_REAL_OUTPUT);
  dr_flag = true;
  w2c = LoadW2C("w2c.txt");
  TrackerInit(rgb_mat);

#endif

}

void ColorAttributesTracker::TrackerInit(Mat img) {

  xo_npca = Mat();
  xo_pca = Mat();

  GetSubwindow(img, init_pos_, sz, non_compressed_features, compressed_features,
               w2c, &xo_npca, &xo_pca);
  z_npca = xo_npca;
  z_pca = xo_pca;
  num_compressed_dim = min(num_compressed_dim, xo_pca.cols);

  //  % project the features of the new appearance example using the new projection matrix

  if (dr_flag == true)
    DimensionReductionInit();
  Mat x;
  FeatureProjection(xo_npca, xo_pca, projection_matrix, cos_window, x);

  //% calculate the new classifier coefficients
  Mat kernel = DenseGaussKernel(sigma, x, x);

  Mat kf;
  idft(kernel, kf, DFT_COMPLEX_OUTPUT);

  multiply(yf, kf, new_alphaf_num);

  multiply(kf, (kf + lambda), new_alphaf_den);  //  new_alphaf_num = yf .* kf;  new_alphaf_den = kf .* (kf + lambda);

  //% first frame, train with a single image
  alphaf_num = new_alphaf_num;
  alphaf_den = new_alphaf_den;

}

void ColorAttributesTracker::Update(Mat img) {

  Mat zp;

  FeatureProjection(z_npca, z_pca, projection_matrix, cos_window, zp);

  //extract the feature map of the local image patch
  GetSubwindow(img, pos, sz, non_compressed_features, compressed_features, w2c,
               &xo_npca, &xo_pca);
  //do the dimensionality reduction and windowing
  Mat x;
  FeatureProjection(xo_npca, xo_pca, projection_matrix, cos_window, x);

  //calculate the response of the classifier
  Mat kernel = DenseGaussKernel(sigma, x, zp);
  Mat kf;

  dft(kernel, kf, DFT_REAL_OUTPUT);
  Mat a, b, response;

//  printf("FeatureProjection %d %d %d\n",alphaf_num.cols,alphaf_num.rows,alphaf_num.channels());
//  printf("FeatureProjection %d %d %d\n",kf.cols,kf.rows,kf.channels());
  multiply(alphaf_num, kf, a);
  divide(a, alphaf_den, b);
  idft(b, response, DFT_REAL_OUTPUT);
  Point maxLoc;
  minMaxLoc(response, NULL, NULL, NULL, &maxLoc);
  pos.x = pos.x - floor(sz.width / 2) + maxLoc.x;
  pos.y = pos.y - floor(sz.height / 2) + maxLoc.y;

  GetSubwindow(img, pos, sz, non_compressed_features, compressed_features, w2c,
               &xo_npca, &xo_pca);

  //% update the appearance
  z_npca = (1 - learning_rate) * z_npca + learning_rate * xo_npca;
  z_pca = (1 - learning_rate) * z_pca + learning_rate * xo_pca;

  ///use_dimensionality_reduction
  if (dr_flag == true)
    DimensionReduction();
  ///use_dimensionality_reduction END

  FeatureProjection(xo_npca, xo_pca, projection_matrix, cos_window, x);

  //% calculate the new classifier coefficients
  kernel = DenseGaussKernel(sigma, x, x);

  dft(kernel, kf, DFT_REAL_OUTPUT);
  multiply(yf, kf, new_alphaf_num);

  multiply(kf, (kf + lambda), new_alphaf_den);  //  new_alphaf_num = yf .* kf;  new_alphaf_den = kf .* (kf + lambda);

  // % subsequent frames, update the model
  alphaf_num = (1 - learning_rate) * alphaf_num
      + learning_rate * new_alphaf_num;
  alphaf_den = (1 - learning_rate) * alphaf_den
      + learning_rate * new_alphaf_den;


}

void ColorAttributesTracker::FeatureProjection(const Mat x_npca,
                                               const Mat x_pca,
                                               const Mat projection_matrix,
                                               const Mat cos_window, Mat& z) {
  vector<Mat> mat_vect;
  if (x_pca.empty()) {
    z = x_npca;
    multiply(x_npca, cos_window, z);   //bug TO-DO
  } else {
    int height = cos_window.rows;
    int width = cos_window.cols;

    //printf("FeatureProjection %d %d %d\n",x_pca.cols,x_pca.rows,x_pca.channels());
    //Mat tmp = x_pca * projection_matrix;
    Mat tmp = pca_.project(x_pca);
    Mat x_proj_pca = tmp.reshape(10, height);  //TO-DO
    //printf("FeatureProjection %d %d %d\n",x_proj_pca.cols,x_proj_pca.rows,x_proj_pca.channels());
    if (x_npca.empty()) {
      z = x_proj_pca;
    } else {
      vector<Mat> mat_vect;

      split(x_proj_pca, mat_vect);

      //mat_vect.insert(mat_vect.begin(), x_npca);

      Mat t;

      for (unsigned int i = 0; i < mat_vect.size(); ++i) {
        //printf("FeatureProjection %d %d %d\n",mat_vect[i].cols,mat_vect[i].rows,mat_vect[i].channels());
        //printf("FeatureProjection %d %d %d\n",cos_window.cols,cos_window.rows,cos_window.channels());
        multiply(mat_vect[i], cos_window, mat_vect[i]);

      }
      merge(mat_vect, z);
    }
  }
}

Mat ColorAttributesTracker::DenseGaussKernel(const double sigma, const Mat x,
                                             const Mat y) {
//% k = dense_gauss_kernel(sigma, x, y)
//%
//% Computes the kernel output for multi-dimensional feature maps x and y
//% using a Gaussian kernel with standard deviation sigma.

  Mat xf = Mat::zeros(x.size(), CV_32FC1);
  vector<Mat> x_v, xf_v;
  split(x, x_v);

  for (int i = 0; i < x.channels(); ++i) {
    Mat tmp;
    dft(x_v[0], tmp, DFT_COMPLEX_OUTPUT);
    xf_v.push_back(tmp);
  }
  merge(xf_v, xf);
  double xx = norm(x);  //squared norm of x

  Mat yf;

  vector<Mat> y_v, yf_v;
  split(y, y_v);
  //dft(y, yf, DFT_COMPLEX_OUTPUT);
  for (int i = 0; i < y.channels(); ++i) {
    Mat tmp;
    dft(y_v[0], tmp, DFT_COMPLEX_OUTPUT);
    yf_v.push_back(tmp);
  }
  merge(yf_v, yf);
  double yy = norm(y);

  //%cross-correlation term in Fourier domain
  vector<Mat> planes;
  Mat zero_mat = Mat::zeros(yf.size(), CV_32F);
  split(yf, planes);
  Mat conj_yf;
  planes[1] = zero_mat - planes[1];
  merge(planes, conj_yf);

  Mat xyf;
//  printf("%d %d %d\n",xf.rows,xf.cols,xf.channels());
//  printf("%d %d %d\n",conj_yf.rows,conj_yf.cols,conj_yf.channels());
  multiply(xf, conj_yf, xyf);

  Mat split_planes[] = { Mat::zeros(xyf.size(), CV_32F), Mat::zeros(xyf.size(),
  CV_32F), Mat::zeros(xyf.size(), CV_32F) };

  split_planes[0] += planes[0] + planes[2];
  Mat xy;
  idft(split_planes[0], xy);

//  printf("FeatureProjection %d %d %d\n",alphaf_num.cols,alphaf_num.rows,alphaf_num.channels());
//  printf("FeatureProjection %d %d %d\n",kf.cols,kf.rows,kf.channels());

  int num = x.dims;
  for (int i = 0; i < xy.rows; ++i)
    for (int j = 0; j < xy.cols; ++j) {
      float t = std::max(0.0, (xx + yy - 2 * xy.at<float>(i, j)));
      t = -1.0 * t / num;
      xy.at<float>(i, j) += exp(t);
    }

  return xy;
}

void ColorAttributesTracker::GetSubwindow(const Mat im, Point pos, Size sz,
                                          const Mat non_pca_features,
                                          const Mat pca_features, const Mat w2c,
                                          Mat* out_npca, Mat* out_pca) {
//  % [out_npca, out_pca] = get_subwindow(im, pos, sz, non_pca_features, pca_features, w2c)
//  %
//  % Extracts the non-PCA and PCA features from image im at position pos and
//  % window size sz. The features are given in non_pca_features and
//  % pca_features. out_npca is the window of non-PCA features and out_pca is
//  % the PCA-features reshaped to [prod(sz) num_pca_feature_dim]. w2c is the
//  % Color Names matrix if used.

  //printf("%d %d\n",pos.x)
  int width = im.cols;
  int height = im.rows;

  Rect roi_rect;
  roi_rect.x = pos.x - sz.width / 2;
  roi_rect.y = pos.y - sz.height / 2;

  //  %check for out-of-bounds coordinates, and set them to the values at
  //  %the borders
  if (roi_rect.x < 0)
    roi_rect.x = 0;

  if (roi_rect.y < 0)
    roi_rect.y = 0;

  if (roi_rect.x + sz.width > width)
    roi_rect.width = width - roi_rect.x;
  else
    roi_rect.width = sz.width;

  if (roi_rect.y + sz.height > height)
    roi_rect.height = height - roi_rect.y;
  else
    roi_rect.height = sz.height;

  Mat im_patch = im(roi_rect);
  vector<Mat> feature_map;
  //% compute non-pca feature map
  if (non_pca_features.empty()) {
    out_npca = NULL;
  }
  else {
    feature_map = GetFeatureMap(im_patch, cn, w2c);
    merge(feature_map, (*out_npca));
  }
//  % compute pca feature map
  if (pca_features.empty()) {
    out_pca = NULL;
  } else {
    feature_map = GetFeatureMap(im_patch, cn, w2c);

    Mat temp_pca, temp_pca_t;

    for (unsigned int i = 0; i < feature_map.size(); ++i) {
      feature_map[i] = feature_map[i].reshape(1, sz.width * sz.height);
    }
    merge(feature_map, temp_pca);
    *out_pca = temp_pca.reshape(1, sz.width * sz.height);
  }

}

vector<Mat> ColorAttributesTracker::GetFeatureMap(Mat im_patch,
                                                  FEATURES features, Mat w2c) {
  vector<Mat> outs;
  Mat im_gray_patch;
  cvtColor(im_patch, im_gray_patch, CV_RGB2GRAY);
//	int feature_levels[2] = {1,10};//the dimension of the valid features
//	int level = 0;
//	int num_feature_levels = feature_levels[(int)features];
  Mat out;
  if (im_patch.channels() == 1) {
    out = Mat_<float>::zeros(im_patch.rows, im_patch.cols);
    out = im_gray_patch / 255 - 0.5;
    outs.push_back(out);
  } else {
    out = Mat_<float>::zeros(im_patch.rows, im_patch.cols);
    out = im_gray_patch / 255 - 0.5;
    outs.push_back(out);
    IM2C(outs, im_patch, w2c, -2);
  }

  return outs;
}
void ColorAttributesTracker::IM2C(vector<Mat>&outs, Mat im_patch, Mat w2c,
                                  int color) {
//	float color_values[11][3] = { { 0, 0, 0 }, { 0, 0, 1 }, { .5, .4, .25 }, { .5, .5, .5 }, { 0, 1, 0 }, { 1, .8, 0 }, { 1, .5, 1 }, { 1, 0, 1 }, { 1, 0, 0 },
//			{ 1, 1, 1 }, { 1, 1, 0 } };
  Mat BB, GG, RR;
  outs.clear();
  vector<Mat> channels;
  split(im_patch, channels);
  BB = channels.at(0);
  GG = channels.at(1);
  RR = channels.at(2);
  vector<int> index_im;
  MatIterator_<uchar> itBB, endBB, itGG, endGG, itRR, endRR;
  for (itBB = BB.begin<uchar>(), endBB = BB.end<uchar>(), itGG =
      GG.begin<uchar>(), itRR = RR.begin<uchar>(); itBB != endBB;
      ++itBB, ++itGG, ++itRR) {
    int index = 1 + floor(*itRR / 8) + 32 * floor(*itGG / 8)
        + 32 * 32 * floor(*itBB / 8);
    //printf("*itRR = %d *itGG = %d  index = %d\n",*itRR,*itGG,index);
    index_im.push_back(index);
  }

  for (int k = 0; k < 10; ++k) {
    Mat out = Mat_<float>::zeros(im_patch.rows, im_patch.cols);
    int index = 0;
    for (int i = 0; i < im_patch.cols; ++i) {
      for (int j = 0; j < im_patch.rows; ++j) {
        out.at<float>(j, i) = w2c.at<float>(index_im[index++], k);
      }
    }
    outs.push_back(out);
  }
}

Mat ColorAttributesTracker::LoadW2C(char* file_name) {
  FILE* fp;
  fp = fopen(file_name, "r");
  if (fp == NULL) {
    printf("Cannot find w2c.txt!\n");
    exit(-1);
  }
  Mat w2c = Mat_<float>::zeros(32768, 10);
  for (int i = 0; i < 32768; ++i) {
    for (int j = 0; j < 10; ++j) {
      float value;
      int flag = fscanf(fp, "%f", &value);
      if (flag < 0) {
        printf("Wrong File!\n");
        exit(-1);
      }
      w2c.at<float>(i, j) = value;
    }
  }
  fclose(fp);
  return w2c;
}

void ColorAttributesTracker::DimensionReduction() {
  //Scalar data_mean = cv::mean(z_pca);
//	vector<float> data_mean;
//	for (int i = 0; i < z_pca.cols; ++i) {
//		float sum = .0f;
//		for (int j = 0; j < z_pca.rows; ++j)
//			sum += z_pca.at<float>(j, i);
//		data_mean.push_back(sum / z_pca.rows);
//		for (int j = 0; j < z_pca.rows; ++j)
//			z_pca.at<float>(j, i) -= data_mean[i];
//	}
//	Mat cov_matrix;
//	cov_matrix = 1 / (sz.height * sz.width - 1) * (z_pca.t() * z_pca);
//	cov_matrix *= compression_learning_rate;
//	cov_matrix += (1 - compression_learning_rate) * old_cov_matrix;
//	SVD thissvd(cov_matrix, SVD::FULL_UV);
//	Mat pca_basis = thissvd.u;
//	Mat pca_variances = thissvd.w;
//	projection_matrix = pca_basis(Rect(0, 0, num_compressed_dim, pca_basis.rows));
//	projection_variances = pca_variances(Rect(0, 0, num_compressed_dim, num_compressed_dim));
//	old_cov_matrix = (1 - compression_learning_rate) * old_cov_matrix
//			+ compression_learning_rate * (projection_matrix * projection_variances * projection_matrix.t());

  int max_components = 10;
  Mat cov_matrix = 1 / (sz.height * sz.width - 1) * (z_pca.t() * z_pca);
  cov_matrix *= compression_learning_rate;
  cov_matrix += (1 - compression_learning_rate) * old_cov_matrix;
  pca_ = PCA(cov_matrix,  // pass the data
      Mat(),  // there is no pre-computed mean vector,
              // so let the PCA engine to compute it
      CV_PCA_DATA_AS_ROW,  // indicate that the vectors
                           // are stored as matrix rows
                           // (use CV_PCA_DATA_AS_COL if the vectors are
                           // the matrix columns)
      max_components  // specify how many principal components to retain
      );

  old_cov_matrix = (1 - compression_learning_rate) * old_cov_matrix
      + compression_learning_rate * pca_.backProject(cov_matrix);
}

void ColorAttributesTracker::DimensionReductionInit() {
//	vector<float> data_mean;
//	for (int i = 0; i < z_pca.cols; ++i) {
//		float sum = .0f;
//		for (int j = 0; j < z_pca.rows; ++j)
//			sum += z_pca.at<float>(j, i);
//		data_mean.push_back(sum / z_pca.rows);
//		for (int j = 0; j < z_pca.rows; ++j)
//			z_pca.at<float>(j, i) -= data_mean[i];
//	}
//	Mat cov_matrix;
//  printf("DimensionReductionInit %d %d %d\n",z_pca.cols,z_pca.rows,z_pca.channels());
//	cov_matrix = 1 / (sz.height * sz.width - 1) * (z_pca.t() * z_pca);
//	SVD thissvd(cov_matrix, SVD::FULL_UV);
//	Mat pca_basis = thissvd.u;
//	Mat pca_variances = thissvd.w;
//  printf("DimensionReductionInit %d %d %d %d %d\n",pca_variances.cols,pca_variances.rows,pca_variances.channels(),num_compressed_dim,pca_basis.rows);
//	projection_matrix = pca_basis(Rect(0, 0, num_compressed_dim, pca_basis.rows));
//
//	projection_variances = pca_variances(Rect(0, 0, num_compressed_dim, num_compressed_dim));
//
//	old_cov_matrix = projection_matrix * projection_variances * projection_matrix.t();

  int max_components = 10;
  Mat cov_matrix = 1 / (sz.height * sz.width - 1) * (z_pca.t() * z_pca);
  pca_ = PCA(cov_matrix,  // pass the data
      Mat(),  // there is no pre-computed mean vector,
              // so let the PCA engine to compute it
      CV_PCA_DATA_AS_ROW,  // indicate that the vectors
                           // are stored as matrix rows
                           // (use CV_PCA_DATA_AS_COL if the vectors are
                           // the matrix columns)
      max_components  // specify how many principal components to retain
      );

  old_cov_matrix = pca_.backProject(cov_matrix);

}
