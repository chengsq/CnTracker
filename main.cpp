/*
 * main.cpp
 *
 *  Created on: Nov 21, 2014
 *      Author: shiqingcheng
 */

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "color_attributes_tracker.h"

#include <iostream>
#include <ctype.h>
#include "config.h"
#include <stdio.h>

using namespace cv;
using namespace std;
enum Status {success,failed};

Status ReadRawData(char* file_name, void* data, int size, int n);

////////////////////////////////////////////////////

Status LoadGroundTruth(char* file_name, vector<Rect>& ground_truth,
		int line_num) {

	FILE* fp = fopen(file_name, "r");
	ground_truth.clear();
	if (fp == NULL){
		printf("[%s] file does not exist!\n", file_name);
		return failed;
	}

	Rect tmp;
	for (int i = 0; i < line_num; ++i) {
		fscanf(fp, "%d,%d,%d,%d", &tmp.x, &tmp.y, &tmp.width, &tmp.height);
		ground_truth.push_back(tmp);
	}

	fclose(fp);
	return success;
}

Status LoadFrameConfig(char *file_name,int* start_frame,int *end_frame)
{
	FILE* fp = fopen(file_name, "r");

	if (fp) {
		fscanf(fp, "%d,%d", start_frame, end_frame);
		fclose(fp);
		return success;
	} else {
		printf("[%s] file does not exist!\n", file_name);
		return failed;
	}
}


void TrackerEvaluation(vector<Rect>& ground_truth,vector<Rect>& tracker_result)
{
	//TO-DO
}

void RunTracker(int start_frame,int end_frame,vector<Rect>& ground_truth,char* image_path)
{
	Point init_pos;
	Point init__center_pos;
	Size target_sz;
	char window_name[100] = "cn_tracker";
	Rect target_rect = ground_truth[0];
	init_pos.x = ground_truth[0].x;
	init_pos.y = ground_truth[0].y;
	target_sz.width = ground_truth[0].width;
	target_sz.height = ground_truth[0].height;
	init__center_pos.x = init_pos.x + target_sz.width / 2;
	init__center_pos.y = init_pos.y + target_sz.height / 2;

	Mat im_mat;
	ColorAttributesTracker* cn_tracker;
	namedWindow(window_name, 1);
	char image_name[1024];
	for (int i = start_frame; i < end_frame; ++i) {
		sprintf(image_name, "%s%06d.jpg", image_path, i);
		im_mat = imread(image_name, CV_LOAD_IMAGE_COLOR);

		if (i == start_frame) {
			cn_tracker = new ColorAttributesTracker(im_mat, init__center_pos.x,
					init__center_pos.y, target_sz.width, target_sz.height, 0);
		} else {
			cn_tracker->Update(im_mat);
			init__center_pos = cn_tracker->pos;
			target_rect.x = init__center_pos.x - target_rect.width / 2;
			target_rect.y = init__center_pos.y - target_rect.height / 2;
		}

		circle(im_mat, init__center_pos, 4, Scalar(255, 255, 0));
		rectangle(im_mat, target_rect, Scalar(0, 255, 255), 1);
		rectangle(im_mat, ground_truth[i-start_frame], Scalar(0, 0, 255), 2);
		imshow(window_name, im_mat);
		waitKey(1);
	}

	imshow(window_name, im_mat);
	waitKey(1);
	destroyWindow(window_name);
}

void UpdateRgbMat(Mat &vis, unsigned char *rgb_data,bool* fg_mask,int width,
                                  int height) {
  unsigned char *rgb_buf = rgb_data;

  for (int i = 0; i < height; i++) {
    unsigned char *Mi;
    Mi = vis.ptr<unsigned char>(i);
    for (int j = 0; j < width; j++) {
      Mi[3 * j + 0] = rgb_buf[3 * j + 2];
      Mi[3 * j + 1] = rgb_buf[3 * j + 1];
      Mi[3 * j + 2] = rgb_buf[3 * j + 0];

      if (fg_mask != NULL) {
        if (fg_mask[i * width + j] == false) {
          Mi[3 * j + 0] = 255;
          Mi[3 * j + 1] = 255;
          Mi[3 * j + 2] = 255;
        }
      }
    }
    rgb_buf += width * 3;
  }
}
void RunTracker(int start_frame,int end_frame,vector<Rect>& ground_truth,Config& conf)
{
	Point init_pos;
	Point init__center_pos;
	Size target_sz;
	char window_name[100] = "cn_tracker";
	Rect target_rect = ground_truth[0];
	init_pos.x = ground_truth[0].x;
	init_pos.y = ground_truth[0].y;
	target_sz.width = ground_truth[0].width;
	target_sz.height = ground_truth[0].height;
	init__center_pos.x = init_pos.x + target_sz.width / 2;
	init__center_pos.y = init_pos.y + target_sz.height / 2;

	ColorAttributesTracker* cn_tracker;
	namedWindow(window_name, 1);
	char image_name[1024];
	int h = 480;
	int w = 640;
	Mat im_mat = Mat(h,w,CV_8UC3);
	unsigned char *rgb = new unsigned char[h * w * 3];
	unsigned short *depth = new unsigned short[h * w];
	bool* mask = new bool[h * w];

	char rgb_name[1024];
	char depth_name[1024];
	char mask_name[1024];
	char img_dir[1024];
	sprintf(img_dir, "%s/%s/%s", conf.sequence_base_path.c_str(),
			conf.sequence_name.c_str(), conf.image_dir.c_str());
	for (int i = start_frame; i < end_frame; ++i) {

		sprintf(rgb_name, "%s/rgb-%06d.dat", img_dir,i);
		sprintf(depth_name, "%s/depth-%06d.dat", img_dir,i);
		sprintf(mask_name, "%s/fgmask-%06d.dat", img_dir,i);

		ReadRawData(rgb_name, rgb, sizeof(char), h * w * 3);
		ReadRawData(depth_name, depth, sizeof(short), h * w);
		ReadRawData(mask_name, mask, sizeof(bool), h * w);
		UpdateRgbMat(im_mat, rgb, mask, w, h);

		if (i == start_frame) {
			cn_tracker = new ColorAttributesTracker(im_mat, init__center_pos.x,
					init__center_pos.y, target_sz.width, target_sz.height, 0);
		} else {
			cn_tracker->Update(im_mat);
			init__center_pos = cn_tracker->pos;
			target_rect.x = init__center_pos.x - target_rect.width / 2;
			target_rect.y = init__center_pos.y - target_rect.height / 2;
		}

		circle(im_mat, init__center_pos, 4, Scalar(255, 255, 0));
		rectangle(im_mat, target_rect, Scalar(0, 255, 255), 1);
		rectangle(im_mat, ground_truth[i - start_frame], Scalar(0, 0, 255), 2);
		imshow(window_name, im_mat);
		waitKey(1);
	}
	delete[] rgb;
	delete[] depth;
	delete[] mask;
}


Status ReadRawData(char* file_name, void* data, int size, int n) {
	FILE* fp = fopen(file_name, "r");
	short resolution[2];
	if (fp) {
		fread(resolution,sizeof(short),2,fp);
		fread(data, size, n, fp);
		printf("ReadRawData");
		fclose(fp);
		return success;
	}
	return failed;
}

int main(int argc, char *argv[]) {
//  freopen("name.txt","w",stdout);
	string configPath = "config.txt";
	if (argc > 1) {
		configPath = argv[1];
	}
	Config conf(configPath);
	cout << conf << endl;

	char sequence_dir[1024];
	sprintf(sequence_dir, "%s/%s", conf.sequence_base_path.c_str(),
			conf.sequence_name.c_str());

	char frame_file_name[1024];
	sprintf(frame_file_name, "%s/%s_frames.txt", sequence_dir,
			conf.sequence_name.c_str());

	int start_frame;
	int end_frame;
	Status status;
	status = LoadFrameConfig(frame_file_name, &start_frame, &end_frame);

	if (status == failed)
		return 0;

	char gt_file_name[1024];
	sprintf(gt_file_name, "%s/%s_gt.txt", sequence_dir,
			conf.sequence_name.c_str());

	vector<Rect> ground_truth;

	status = LoadGroundTruth(gt_file_name, ground_truth,
			end_frame - start_frame + 1);

	if (status == failed)
		return 0;

	char image_path[1024];
	sprintf(image_path, "%s/%s/%s", sequence_dir,
					conf.image_dir.c_str(), conf.image_prefix.c_str());
	RunTracker(start_frame,end_frame,ground_truth,image_path);
	//RunTracker(start_frame,end_frame,ground_truth,conf);
	return 0;
}


