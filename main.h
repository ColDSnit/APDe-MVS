#ifndef _MAIN_H_
#define _MAIN_H_
// Ensure std::launch is declared for Boost/NVCC host compilation path
#include <future>
// Includes Opencv
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
// #include <cuda_texture_types.h>  // Removed for CUDA 12.8+ Windows compatibility
#include <curand_kernel.h>
#include <vector_types.h>
// Includes STD libs
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <cstdarg>
#include <random>
#include <unordered_map>
// Includes Boost filesystem
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
// Includes ThreadPool
#include "ThreadPool.h"

// Define some const var
#define MAX_IMAGES 32
#define ANCHOR_NUM 9
#define MAX_SEARCH_RADIUS 4096
#define DEBUG_POINT_X 753
#define DEBUG_POINT_Y 259
#define RELIABLE_CURVE_SAMPLE_NUM 61


using namespace boost::filesystem;

struct Camera {
    float K[9];
    float R[9];
    float t[3];
    float c[3];
    int height;
    int width;
    float depth_min;
    float depth_max;
    float interval;
    float depth_num;
};

struct PointList {
    float3 coord;
    float3 color;
};

enum RunState {
    FIRST_INIT,
    REFINE_INIT,
    REFINE_ITER,
};

enum PixelState {
    WEAK,
    STRONG,
    UNKNOWN
};

struct PatchMatchParams {
    int max_iterations = 3;
    int num_images = 5;
    int top_k = 4;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    bool geom_consistency = false;
    bool use_impetus = true;
    int strong_radius = 5;
    int strong_increment = 2;
    int weak_radius = 5;
    int weak_increment = 5;
    bool use_APD = true;
    bool use_sa = true;
    int weak_peak_radius = 2;
    int rotate_time = 4;
    float ransac_threshold = 0.005;
    float geom_factor = 0.2f; // eth
     // float geom_factor = 0.05f; // tat
    RunState state;
};

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
    path dense_folder;
    path result_folder;
    int scale_size = 1;
    PatchMatchParams params;
    bool show_medium_result = false;
    bool export_anchor = false;
    bool export_reliable_curve = false;
    int iteration;
    std::string img_ext;
    int used_time;
};

#endif // !_MAIN_H_
