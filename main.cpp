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
#include <stdio.h>


char suqences_dir[100] = "sequences";
char video_dir[100] = "deer";
char images_dir[100] =  "imgs";
char gt_name[100] ="deer_gt.txt";
char image_pre[100] ="img";

int frame_cnt = 71;

using namespace cv;
using namespace std;

char image_name[1024];
char window_name[100] = "cn_tracker";
char gt_file_name[1024];
Point init_pos;
Point init__center_pos;
Size  target_sz;
Rect target_rect;

ColorAttributesTracker* cn_tracker;
int main()
{

  Mat im_mat;
  namedWindow(window_name, 1);
  sprintf(gt_file_name,"%s/%s/%s",suqences_dir,video_dir,gt_name);
  //printf("%s\n",gt_file_name);

  FILE* fp = fopen(gt_file_name,"r");

  if (fp) {
    fscanf(fp, "%d,%d,%d,%d", &init_pos.x, &init_pos.y, &target_sz.width,
           &target_sz.height);
    //printf("init_pos: %d,%d,%d,%d\n",init_pos.x, init_pos.y,target_sz.width,target_sz.height);
    init__center_pos.x = init_pos.x + target_sz.width / 2;
    init__center_pos.y = init_pos.y + target_sz.height / 2;
    target_rect.x = init_pos.x;
    target_rect.y = init_pos.y;
    target_rect.width = target_sz.width;
    target_rect.height = target_sz.height;
  }


  for(int i = 0;i < frame_cnt-1;++i)
  {
    sprintf(image_name,"%s/%s/%s/%s%05d.jpg",suqences_dir,video_dir,images_dir,image_pre,i+1);
   // printf("%s\n",image_name);
    im_mat = imread(image_name, CV_LOAD_IMAGE_COLOR);
    if(i == 0)
    {
      cn_tracker = new ColorAttributesTracker(im_mat,init__center_pos.y,init__center_pos.y,target_sz.width,target_sz.height,0);
    }
    else{
      cn_tracker->Update(im_mat);
      init__center_pos = cn_tracker->pos;
      target_rect.x = init__center_pos.x - target_rect.width / 2;
      target_rect.y = init__center_pos.y - target_rect.height / 2;
    }

    circle(im_mat,init__center_pos,3,Scalar(0,255,0));
    rectangle(im_mat, target_rect, Scalar(0,255,255), 1);
    imshow(window_name,im_mat);
    waitKey(0);
  }
  destroyWindow(window_name);
}



