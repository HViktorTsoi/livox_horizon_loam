// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               Livox@gmail.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <ceres/ceres.h>
#include <geometry_msgs/PoseStamped.h>
#include <loam_horizon/common.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "lidarFactor.hpp"
#include "loam_horizon/common.h"
#include "loam_horizon/tic_toc.h"

// 这部分代码是求解当前帧在map世界坐标系下的绝对位姿，一共维护3组变量
// q_w_curr: 待优化变量，是当前帧在世界坐标系下的绝对位姿
// q_wmap_wodom: 是世界坐标系下地图到里程计之间的相对位姿, 维护这个变量纯粹是为了方便，因为优化出w_curr并且给定了odometry,这个量是直接能算出来的
// q_wodom_curr: 是通过laserOdometry帧间匹配得到的里程计，直接拿过来作为原始观测量用了，在这部分代码里没有被更改
// t与q的含义相同
int frameCount = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;

const int laserCloudNum =
    laserCloudWidth * laserCloudHeight * laserCloudDepth;  // 4851

int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(
    new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(
    new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(
    new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(
    new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(
    new pcl::PointCloud<PointType>());

// input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(
    new pcl::PointCloud<PointType>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudFullResColor(
    new pcl::PointCloud<pcl::PointXYZRGB>());

// points in every cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

// kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(
    new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(
    new pcl::KdTreeFLANN<PointType>());

double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes,
    pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;

// set initial guess
void transformAssociateToMap() {
  q_w_curr = q_wmap_wodom * q_wodom_curr;
  t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void transformUpdate() {
  q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
  t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void pointAssociateToMap(PointType const *const pi, PointType *const po) {
  Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
  Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
  po->x = point_w.x();
  po->y = point_w.y();
  po->z = point_w.z();
  po->intensity = pi->intensity;
  // po->intensity = 1.0;
}

void RGBpointAssociateToMap(PointType const *const pi,
                            pcl::PointXYZRGB *const po) {
  Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
  Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
  po->x = point_w.x();
  po->y = point_w.y();
  po->z = point_w.z();
  int reflection_map = pi->curvature * 10;
  if (reflection_map < 30) {
    int green = (reflection_map * 255 / 30);
    po->r = 0;
    po->g = green & 0xff;
    po->b = 0xff;
  } else if (reflection_map < 90) {
    int blue = (((90 - reflection_map) * 255) / 60);
    po->r = 0x0;
    po->g = 0xff;
    po->b = blue & 0xff;
  } else if (reflection_map < 150) {
    int red = ((reflection_map - 90) * 255 / 60);
    po->r = red & 0xff;
    po->g = 0xff;
    po->b = 0x0;
  } else {
    int green = (((255 - reflection_map) * 255) / (255 - 150));
    po->r = 0xff;
    po->g = green & 0xff;
    po->b = 0;
  }
}

void pointAssociateTobeMapped(PointType const *const pi, PointType *const po) {
  Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
  Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
  po->x = point_curr.x();
  po->y = point_curr.y();
  po->z = point_curr.z();
  po->intensity = pi->intensity;
}

void laserCloudCornerLastHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2) {
  mBuf.lock();
  cornerLastBuf.push(laserCloudCornerLast2);
  mBuf.unlock();
}

void laserCloudSurfLastHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2) {
  mBuf.lock();
  surfLastBuf.push(laserCloudSurfLast2);
  mBuf.unlock();
}

void laserCloudFullResHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2) {
  mBuf.lock();
  fullResBuf.push(laserCloudFullRes2);
  mBuf.unlock();
}

// receive odomtry
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry) {
  mBuf.lock();
  odometryBuf.push(laserOdometry);
  mBuf.unlock();

  // high frequence publish
  Eigen::Quaterniond q_wodom_curr;
  Eigen::Vector3d t_wodom_curr;
  q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
  q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
  q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
  q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
  t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
  t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
  t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

  // 世界到里程计 * 里程计到当前帧
  Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
  Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;

  nav_msgs::Odometry odomAftMapped;
  odomAftMapped.header.frame_id = "/camera_init";
  odomAftMapped.child_frame_id = "/aft_mapped";
  odomAftMapped.header.stamp = laserOdometry->header.stamp;
  odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
  odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
  odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
  odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
  odomAftMapped.pose.pose.position.x = t_w_curr.x();
  odomAftMapped.pose.pose.position.y = t_w_curr.y();
  odomAftMapped.pose.pose.position.z = t_w_curr.z();
  pubOdomAftMappedHighFrec.publish(odomAftMapped);
}

void process() {
  while (1) {
    // 拿数据
    while (!cornerLastBuf.empty() && !surfLastBuf.empty() &&
           !fullResBuf.empty() && !odometryBuf.empty()) {
      mBuf.lock();
      while (!odometryBuf.empty() &&
             odometryBuf.front()->header.stamp.toSec() <
                 cornerLastBuf.front()->header.stamp.toSec())
        odometryBuf.pop();
      if (odometryBuf.empty()) {
        mBuf.unlock();
        break;
      }

      while (!surfLastBuf.empty() &&
             surfLastBuf.front()->header.stamp.toSec() <
                 cornerLastBuf.front()->header.stamp.toSec())
        surfLastBuf.pop();
      if (surfLastBuf.empty()) {
        mBuf.unlock();
        break;
      }

      while (!fullResBuf.empty() &&
             fullResBuf.front()->header.stamp.toSec() <
                 cornerLastBuf.front()->header.stamp.toSec())
        fullResBuf.pop();
      if (fullResBuf.empty()) {
        mBuf.unlock();
        break;
      }

      timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
      timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
      timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
      timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

      if (timeLaserCloudCornerLast != timeLaserOdometry ||
          timeLaserCloudSurfLast != timeLaserOdometry ||
          timeLaserCloudFullRes != timeLaserOdometry) {
        printf("time corner %f surf %f full %f odom %f \n",
               timeLaserCloudCornerLast, timeLaserCloudSurfLast,
               timeLaserCloudFullRes, timeLaserOdometry);
        printf("unsync messeage!");
        mBuf.unlock();
        break;
      }

      laserCloudCornerLast->clear();
      pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
      cornerLastBuf.pop();

      laserCloudSurfLast->clear();
      pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
      surfLastBuf.pop();

      laserCloudFullRes->clear();
      pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
      fullResBuf.pop();

      // 里程计原点到当前的帧的位姿，直接从laserOdometry中取
      q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
      q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
      q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
      q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
      t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
      t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
      t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
      odometryBuf.pop();

      //      while (!cornerLastBuf.empty()) {
      //        //cornerLastBuf.pop();
      //        printf("drop lidar frame in mapping for real time performance
      //        \n");
      //      }

      mBuf.unlock();

      TicToc t_whole;

      // 用上一次mapping得到的map-odometry结果作为初始值，对新得到的odometry进行变换，
      // 从而获得当前帧在世界坐标系下的初始估计值
      transformAssociateToMap();

      // 先把之前的点云保存在10m*10m*10m的立方体中，
      // 若cube中的点与当前帧中的点云有重叠部分就把他们提取出来保存在KD树中。
      // 我们找地图中的点时，要在特征点附近宽为10cm的立方体邻域内搜索
      TicToc t_shift;

      // 当前点云在整个地图的哪个cube中
      int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth;
      int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;
      int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;

      if (t_w_curr.x() + 25.0 < 0) centerCubeI--;
      if (t_w_curr.y() + 25.0 < 0) centerCubeJ--;
      if (t_w_curr.z() + 25.0 < 0) centerCubeK--;

      // 下边这6段又臭又长的代码的作用：
      // 如果取到的子cube在整个大cube的边缘，则将点对应的cube索引,以及整个大cube的索引，
      // 都循环向中心方向挪动一个单位，这样就能在下一步保证取子cube周围的5x5x5邻域时不会取到边缘之外的点了；
      // 这几段代码过于生猛，有没有更好的实现方法？？？
      while (centerCubeI < 3) {
        for (int j = 0; j < laserCloudHeight; j++) {
          for (int k = 0; k < laserCloudDepth; k++) {
            int i = laserCloudWidth - 1;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k];
            for (; i >= 1; i--) {
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i - 1 + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i - 1 + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j +
                                laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeI++;
        laserCloudCenWidth++;
      }

      while (centerCubeI >= laserCloudWidth - 3) {
        for (int j = 0; j < laserCloudHeight; j++) {
          for (int k = 0; k < laserCloudDepth; k++) {
            int i = 0;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k];
            for (; i < laserCloudWidth - 1; i++) {
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + 1 + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + 1 + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j +
                                laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeI--;
        laserCloudCenWidth--;
      }

      while (centerCubeJ < 3) {
        for (int i = 0; i < laserCloudWidth; i++) {
          for (int k = 0; k < laserCloudDepth; k++) {
            int j = laserCloudHeight - 1;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k];
            for (; j >= 1; j--) {
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * (j - 1) +
                                        laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * (j - 1) +
                                      laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j +
                                laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeJ++;
        laserCloudCenHeight++;
      }

      while (centerCubeJ >= laserCloudHeight - 3) {
        for (int i = 0; i < laserCloudWidth; i++) {
          for (int k = 0; k < laserCloudDepth; k++) {
            int j = 0;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k];
            for (; j < laserCloudHeight - 1; j++) {
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * (j + 1) +
                                        laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * (j + 1) +
                                      laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j +
                                laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeJ--;
        laserCloudCenHeight--;
      }

      while (centerCubeK < 3) {
        for (int i = 0; i < laserCloudWidth; i++) {
          for (int j = 0; j < laserCloudHeight; j++) {
            int k = laserCloudDepth - 1;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k];
            for (; k >= 1; k--) {
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight *
                                            (k - 1)];
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight *
                                          (k - 1)];
            }
            laserCloudCornerArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j +
                                laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeK++;
        laserCloudCenDepth++;
      }

      while (centerCubeK >= laserCloudDepth - 3) {
        for (int i = 0; i < laserCloudWidth; i++) {
          for (int j = 0; j < laserCloudHeight; j++) {
            int k = 0;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k];
            for (; k < laserCloudDepth - 1; k++) {
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight *
                                            (k + 1)];
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight *
                                          (k + 1)];
            }
            laserCloudCornerArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j +
                                laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeK--;
        laserCloudCenDepth--;
      }

      int laserCloudValidNum = 0;
      int laserCloudSurroundNum = 0;

      // 取子cube周围的5x5x5邻域，laserCloudValidInd存储的是这个邻域的每一个cube的坐标
      for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
        for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
          for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++) {
            if (i >= 0 && i < laserCloudWidth && j >= 0 &&
                j < laserCloudHeight && k >= 0 && k < laserCloudDepth) {
              laserCloudValidInd[laserCloudValidNum] =
                  i + laserCloudWidth * j +
                  laserCloudWidth * laserCloudHeight * k;
              laserCloudValidNum++;
              laserCloudSurroundInd[laserCloudSurroundNum] =
                  i + laserCloudWidth * j +
                  laserCloudWidth * laserCloudHeight * k;
              laserCloudSurroundNum++;
            }
          }
        }
      }

      // 把邻域中的边缘和平面特征点云取出来，拼接成局部地图
      laserCloudCornerFromMap->clear();
      laserCloudSurfFromMap->clear();
      for (int i = 0; i < laserCloudValidNum; i++) {
        *laserCloudCornerFromMap +=
            *laserCloudCornerArray[laserCloudValidInd[i]];
        *laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
      }
      int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
      int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

      // 把局部地图下采样
      pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(
          new pcl::PointCloud<PointType>());
      downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
      downSizeFilterCorner.filter(*laserCloudCornerStack);
      int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

      pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(
          new pcl::PointCloud<PointType>());
      downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
      downSizeFilterSurf.filter(*laserCloudSurfStack);
      int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

      printf("map prepare time %f ms\n", t_shift.toc());
      printf("map corner num %d  surf num %d \n", laserCloudCornerFromMapNum,
             laserCloudSurfFromMapNum);

      // 保证地图中有足够的观测
      if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50) {
        TicToc t_opt;
        TicToc t_tree;
        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
        printf("build tree time %f ms \n", t_tree.toc());

        // 迭代求解2个epoch
        for (int iterCount = 0; iterCount < 2; iterCount++) {
          // ceres::LossFunction *loss_function = NULL;
          ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
          ceres::LocalParameterization *q_parameterization =
              new ceres::EigenQuaternionParameterization();
          ceres::Problem::Options problem_options;

          // 优化量是当前帧的绝对位姿
          ceres::Problem problem(problem_options);
          problem.AddParameterBlock(parameters, 4, q_parameterization);
          problem.AddParameterBlock(parameters + 4, 3);

          TicToc t_data;
          int corner_num = 0;

          // 当前帧角的点-地图帧的线匹配
          for (int i = 0; i < laserCloudCornerStackNum; i++) {
            pointOri = laserCloudCornerStack->points[i];
            // double sqrtDis = pointOri.x * pointOri.x + pointOri.y *
            // pointOri.y + pointOri.z * pointOri.z;
            // 用上一帧的位姿作为优化前的初始值，把当前点变换到map frame下
            pointAssociateToMap(&pointOri, &pointSel);
            // 从map中找到当前点最近的邻居点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                                pointSearchSqDis);

            // 保证找到的K近邻确实是在较近的半径内
            if (pointSearchSqDis[4] < 1.0) {
              // 下边的过程和laserOdometry.cpp里的角点匹配过程完全一致，只不过目标线换成了角点地图中的线
              std::vector<Eigen::Vector3d> nearCorners;
              Eigen::Vector3d center(0, 0, 0);
              for (int j = 0; j < 5; j++) {
                Eigen::Vector3d tmp(
                    laserCloudCornerFromMap->points[pointSearchInd[j]].x,
                    laserCloudCornerFromMap->points[pointSearchInd[j]].y,
                    laserCloudCornerFromMap->points[pointSearchInd[j]].z);
                center = center + tmp;
                nearCorners.push_back(tmp);
              }
              center = center / 5.0;

              Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
              for (int j = 0; j < 5; j++) {
                Eigen::Matrix<double, 3, 1> tmpZeroMean =
                    nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
              }

              Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

              // if is indeed line feature
              // note Eigen library sort eigenvalues in increasing order
              Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
              Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
              if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b;
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;

                ceres::CostFunction *cost_function =
                    LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                problem.AddResidualBlock(cost_function, loss_function,
                                         parameters, parameters + 4);
                corner_num++;
              }
            }
            /*
            else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
            {
                    Eigen::Vector3d center(0, 0, 0);
                    for (int j = 0; j < 5; j++)
                    {
                            Eigen::Vector3d
            tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
                                                                    laserCloudCornerFromMap->points[pointSearchInd[j]].y,
                                                                    laserCloudCornerFromMap->points[pointSearchInd[j]].z);
                            center = center + tmp;
                    }
                    center = center / 5.0;
                    Eigen::Vector3d curr_point(pointOri.x, pointOri.y,
            pointOri.z);
                    ceres::CostFunction *cost_function =
            LidarDistanceFactor::Create(curr_point, center);
                    problem.AddResidualBlock(cost_function, loss_function,
            parameters, parameters + 4);
            }
            */
          }

          // 当前帧的平面特征点-地图帧的平面匹配
          int surf_num = 0;
          for (int i = 0; i < laserCloudSurfStackNum; i++) {
            // 这部分的点面匹配和laserOdometry.cpp中的实现也基本一致，只不过cost function变为LidarPlaneNormFactor
            // LidarPlaneNormFactor实际上与laserOdometry.cpp里的LidarPlaneFactor本质上是一致的，就是最小化点-面距离
            pointOri = laserCloudSurfStack->points[i];
            // double sqrtDis = pointOri.x * pointOri.x + pointOri.y *
            // pointOri.y + pointOri.z * pointOri.z;
            pointAssociateToMap(&pointOri, &pointSel);
            // 在地图中找到当前点的近邻点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                              pointSearchSqDis);

            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 =
                -1 * Eigen::Matrix<double, 5, 1>::Ones();
            if (pointSearchSqDis[4] < 1.0) {
              for (int j = 0; j < 5; j++) {
                matA0(j, 0) =
                    laserCloudSurfFromMap->points[pointSearchInd[j]].x;
                matA0(j, 1) =
                    laserCloudSurfFromMap->points[pointSearchInd[j]].y;
                matA0(j, 2) =
                    laserCloudSurfFromMap->points[pointSearchInd[j]].z;
                // printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j,
                // 2));
              }
              // find the norm of plane
              Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
              double negative_OA_dot_norm = 1 / norm.norm();
              norm.normalize();

              // 检验平面拟合的质量
              // Here n(pa, pb, pc) is unit norm of plane
              bool planeValid = true;
              for (int j = 0; j < 5; j++) {
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(
                        norm(0) *
                            laserCloudSurfFromMap->points[pointSearchInd[j]].x +
                        norm(1) *
                            laserCloudSurfFromMap->points[pointSearchInd[j]].y +
                        norm(2) *
                            laserCloudSurfFromMap->points[pointSearchInd[j]].z +
                        negative_OA_dot_norm) > 0.2) {
                  planeValid = false;
                  break;
                }
              }
              Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
              if (planeValid) {
                ceres::CostFunction *cost_function =
                    LidarPlaneNormFactor::Create(curr_point, norm,
                                                 negative_OA_dot_norm);
                problem.AddResidualBlock(cost_function, loss_function,
                                         parameters, parameters + 4);
                surf_num++;
              }
            }
            /*
            else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
            {
                    Eigen::Vector3d center(0, 0, 0);
                    for (int j = 0; j < 5; j++)
                    {
                            Eigen::Vector3d
            tmp(laserCloudSurfFromMap->points[pointSearchInd[j]].x,
                                                                    laserCloudSurfFromMap->points[pointSearchInd[j]].y,
                                                                    laserCloudSurfFromMap->points[pointSearchInd[j]].z);
                            center = center + tmp;
                    }
                    center = center / 5.0;
                    Eigen::Vector3d curr_point(pointOri.x, pointOri.y,
            pointOri.z);
                    ceres::CostFunction *cost_function =
            LidarDistanceFactor::Create(curr_point, center);
                    problem.AddResidualBlock(cost_function, loss_function,
            parameters, parameters + 4);
            }
            */
          }

          // printf("corner num %d used corner num %d \n",
          // laserCloudCornerStackNum, corner_num);
          // printf("surf num %d used surf num %d \n", laserCloudSurfStackNum,
          // surf_num);

          printf("mapping data assosiation time %f ms \n", t_data.toc());

          TicToc t_solver;
          ceres::Solver::Options options;
          options.linear_solver_type = ceres::DENSE_QR;
          options.max_num_iterations = 10;
          options.minimizer_progress_to_stdout = false;
          options.check_gradients = false;
          options.gradient_check_relative_precision = 1e-4;
          ceres::Solver::Summary summary;
          ceres::Solve(options, &problem, &summary);
          printf("mapping solver time %f ms \n", t_solver.toc());
          std::cout << summary.BriefReport() << std::endl;
          // printf("time %f \n", timeLaserOdometry);
          // printf("corner factor num %d surf factor num %d\n", corner_num,
          // surf_num);
          // printf("result q %f %f %f %f result t %f %f %f\n", parameters[3],
          // parameters[0], parameters[1], parameters[2],
          //	   parameters[4], parameters[5], parameters[6]);
        }
        printf("mapping optimization time %f \n", t_opt.toc());
      } else {
        ROS_WARN("time Map corner and surf num are not enough");
      }
      // 优化得到当前帧在世界坐标系下的绝对位姿之后，结合新得到的odometry，
      // 来更新map-odometry的变换
      transformUpdate();

      // 优化得到当前帧的绝对位姿之后，将当前帧点云的特征更新到角点特征地图上
      TicToc t_add;
      for (int i = 0; i < laserCloudCornerStackNum; i++) {
        // 将当前帧的角点转换到世界坐标系下
        pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

        int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
        int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
        int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

        if (pointSel.x + 25.0 < 0) cubeI--;
        if (pointSel.y + 25.0 < 0) cubeJ--;
        if (pointSel.z + 25.0 < 0) cubeK--;

        // 将当前特征点添加到对应cube的特征点集中
        if (cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 &&
            cubeJ < laserCloudHeight && cubeK >= 0 && cubeK < laserCloudDepth) {
          int cubeInd = cubeI + laserCloudWidth * cubeJ +
                        laserCloudWidth * laserCloudHeight * cubeK;
          laserCloudCornerArray[cubeInd]->push_back(pointSel);
        }
      }

      // 同样的操作 更新平面特征地图
      for (int i = 0; i < laserCloudSurfStackNum; i++) {
        pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

        int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
        int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
        int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

        if (pointSel.x + 25.0 < 0) cubeI--;
        if (pointSel.y + 25.0 < 0) cubeJ--;
        if (pointSel.z + 25.0 < 0) cubeK--;

        if (cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 &&
            cubeJ < laserCloudHeight && cubeK >= 0 && cubeK < laserCloudDepth) {
          int cubeInd = cubeI + laserCloudWidth * cubeJ +
                        laserCloudWidth * laserCloudHeight * cubeK;
          laserCloudSurfArray[cubeInd]->push_back(pointSel);
        }
      }
      printf("add points time %f ms\n", t_add.toc());

      // 将更新的那部分重新进行下采样，这里直接使用之前的laserCloudValidInd的原因，
      // 猜测是因为考虑到当前帧和上一阵的位姿差异不大，所以近邻cube不会相差太多，而且下采样是一个不需要太精确的
      // 操作(少下采样一两个cube不会影响整体的地图效果)，因此就直接使用优化前的邻域cube来进行下采样了，不需要再进行一次邻域cube的查找
      TicToc t_filter;
      for (int i = 0; i < laserCloudValidNum; i++) {
        int ind = laserCloudValidInd[i];

        pcl::PointCloud<PointType>::Ptr tmpCorner(
            new pcl::PointCloud<PointType>());
        downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
        downSizeFilterCorner.filter(*tmpCorner);
        laserCloudCornerArray[ind] = tmpCorner;

        pcl::PointCloud<PointType>::Ptr tmpSurf(
            new pcl::PointCloud<PointType>());
        downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
        downSizeFilterSurf.filter(*tmpSurf);
        laserCloudSurfArray[ind] = tmpSurf;
      }
      printf("filter time %f ms \n", t_filter.toc());

      TicToc t_pub;
      // laserCloudSurround存储的是当前帧邻域的角点特征点云+平面特征点云
      // publish surround map for every 5 frame
      if (frameCount % 5 == 0) {
        laserCloudSurround->clear();
        for (int i = 0; i < laserCloudSurroundNum; i++) {
          // ind: 在哪个cube中
          int ind = laserCloudSurroundInd[i];
          // 将那个cube对应的特征加到laserCloudSurround中
          *laserCloudSurround += *laserCloudCornerArray[ind];
          *laserCloudSurround += *laserCloudSurfArray[ind];
        }

        sensor_msgs::PointCloud2 laserCloudSurround3;
        pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
        laserCloudSurround3.header.stamp =
            ros::Time().fromSec(timeLaserOdometry);
        laserCloudSurround3.header.frame_id = "/camera_init";
        pubLaserCloudSurround.publish(laserCloudSurround3);
      }

      // laserCloudMap存储的是开始建图以来所有帧特征构成的地图
      if (frameCount % 20 == 0) {
        pcl::PointCloud<PointType> laserCloudMap;
        for (int i = 0; i < 4851; i++) {
          laserCloudMap += *laserCloudCornerArray[i];
          laserCloudMap += *laserCloudSurfArray[i];
        }
        sensor_msgs::PointCloud2 laserCloudMsg;
        pcl::toROSMsg(laserCloudMap, laserCloudMsg);
        laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        laserCloudMsg.header.frame_id = "/camera_init";
        pubLaserCloudMap.publish(laserCloudMsg);
      }

      // 最后建出来的地图，注意这里没有保存由原始点云构成的高质量地图，而只是发布了最新一帧点云转换到世界坐标系
      // 后的结果，在rviz里是通过decay time显示整个地图的，这里可以改一下逻辑来保存最后的完整地图，但是可能会有较大的存储开销
      laserCloudFullResColor->clear();
      int laserCloudFullResNum = laserCloudFullRes->points.size();
      for (int i = 0; i < laserCloudFullResNum; i++) {
        pcl::PointXYZRGB temp_point;
        // 将原始点云转换到世界坐标系的绝对位姿下，然后为了可视化将intensity转换为RGB
        RGBpointAssociateToMap(&laserCloudFullRes->points[i], &temp_point);
        laserCloudFullResColor->push_back(temp_point);
      }

      sensor_msgs::PointCloud2 laserCloudFullRes3;
      pcl::toROSMsg(*laserCloudFullResColor, laserCloudFullRes3);
      laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      laserCloudFullRes3.header.frame_id = "/camera_init";
      pubLaserCloudFullRes.publish(laserCloudFullRes3);

      printf("mapping pub time %f ms \n", t_pub.toc());

      printf("whole mapping time %f ms +++++\n", t_whole.toc());

      // 发布最终的绝对位姿，以及绝对位姿的tf
      nav_msgs::Odometry odomAftMapped;
      odomAftMapped.header.frame_id = "/camera_init";
      odomAftMapped.child_frame_id = "/aft_mapped";
      odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
      odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
      odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
      odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
      odomAftMapped.pose.pose.position.x = t_w_curr.x();
      odomAftMapped.pose.pose.position.y = t_w_curr.y();
      odomAftMapped.pose.pose.position.z = t_w_curr.z();
      pubOdomAftMapped.publish(odomAftMapped);

      geometry_msgs::PoseStamped laserAfterMappedPose;
      laserAfterMappedPose.header = odomAftMapped.header;
      laserAfterMappedPose.pose = odomAftMapped.pose.pose;
      laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
      laserAfterMappedPath.header.frame_id = "/camera_init";
      laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
      pubLaserAfterMappedPath.publish(laserAfterMappedPath);

      static tf::TransformBroadcaster br;
      tf::Transform transform;
      tf::Quaternion q;
      transform.setOrigin(tf::Vector3(t_w_curr(0), t_w_curr(1), t_w_curr(2)));
      q.setW(q_w_curr.w());
      q.setX(q_w_curr.x());
      q.setY(q_w_curr.y());
      q.setZ(q_w_curr.z());
      transform.setRotation(q);
      br.sendTransform(tf::StampedTransform(transform,
                                            odomAftMapped.header.stamp,
                                            "/camera_init", "/aft_mapped"));

      frameCount++;
    }
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;

  float lineRes = 0;
  float planeRes = 0;
  nh.param<float>("mapping_line_resolution", lineRes, 0.4);
  nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
  printf("line resolution %f plane resolution %f \n", lineRes, planeRes);
  downSizeFilterCorner.setLeafSize(lineRes, lineRes, lineRes);
  downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);

  ros::Subscriber subLaserCloudCornerLast =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100,
                                             laserCloudCornerLastHandler);

  ros::Subscriber subLaserCloudSurfLast =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100,
                                             laserCloudSurfLastHandler);

  // 接受到odomerty之后，要把odometry变换到map frame下，再发布出来
  ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>(
      "/laser_odom_to_init", 100, laserOdometryHandler);

  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>(
      "/velodyne_cloud_3", 100, laserCloudFullResHandler);

  pubLaserCloudSurround =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);

  pubLaserCloudMap =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);

  pubLaserCloudFullRes =
      nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

  pubOdomAftMapped =
      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);

  pubOdomAftMappedHighFrec =
      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);

  pubLaserAfterMappedPath =
      nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

  for (int i = 0; i < laserCloudNum; i++) {
    laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
    laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
  }

  std::thread mapping_process{process};

  ros::spin();

  return 0;
}
