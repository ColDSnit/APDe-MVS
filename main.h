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
// Includes the host-side option structs (fusion, weak filter, pipeline schedule)
#include "apd_options.h"

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
    // Projection model. 0 = pinhole (upstream). 1 = orthographic (telecentric lens): the image position no
    // longer depends on depth; K keeps its pinhole-equivalent values and the magnification is K / ref_depth.
    int model = 0;
    float ref_depth = 1.0f;  // depth at which the pinhole-equivalent K and the orthographic camera agree
};

enum CameraModel {
    CAMERA_PINHOLE = 0,
    CAMERA_ORTHOGRAPHIC = 1
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
    // ---- run-time configurable kernel constants; every default is the upstream literal ----
    float geom_max_cost = 3.0f;            // cap of the geometric consistency cost (pixels)
    float depth_perturbation = 0.02f;      // relative depth perturbation in hypothesis refinement
    float normal_perturbation = 0.02f;     // normal perturbation in hypothesis refinement (fraction of pi)
    float vs_prior_selected = 0.9f;        // view-selection prior when a neighbour selected the view
    float vs_prior_unselected = 0.1f;      // view-selection prior otherwise
    double vs_cost_thresh_init = 0.8;      // good-cost threshold = init * exp(-iter^2 / decay)
    float vs_cost_thresh_decay = 90.0f;
    float vs_good_sigma = 0.18f;           // weight of a good cost = exp(-cost^2 / sigma)
    float vs_fallback_sigma = 0.32f;       // weight when too few good costs = exp(-threshold^2 / sigma)
    float vs_bad_cost = 1.2f;              // NCC cost above which a hypothesis counts as bad for a view
    int vs_min_good = 2;                   // more than this many good hypotheses are needed
    int vs_max_bad = 3;                    // fewer than this many bad hypotheses are allowed
    int vs_num_samples = 15;               // Monte-Carlo view samples per pixel (<= 255)
    double refine_init_margin = 0.1;       // cost improvement required to accept a change in REFINE_INIT
    double weak_center_weight = 0.25;      // deformable NCC = w * centre patch + (1 - w) * anchor patches
    int border_margin = 6;                 // image border (pixels) never classified or used as anchor
    float weak_max_peak_cost = 0.5f;       // cost-curve minimum above this => WEAK
    float strong_single_peak_cost = 0.15f; // single-peak curve below this => STRONG
    float strong_multi_peak_var = 0.2f;    // multi-peak curve with peak spread above this => STRONG
    int weak_filter_radius = 2;            // isolated-STRONG removal window radius
    float conf_reproj_px = 2.0f;           // confidence: reprojection tolerance (pixels)
    float conf_depth_rel = 0.02f;          // confidence: relative depth tolerance
    int conf_w_exist = 1;                  // confidence votes: source depth exists
    int conf_w_reproj = 2;                 // confidence votes: reprojection test passed
    int conf_w_depth = 2;                  // confidence votes: depth test passed
    bool local_refine = true;              // run the final disparity-step depth refinement
    int local_refine_radius = 5;           // refinement search radius in disparity steps
    double local_refine_margin = 0.1;      // cost improvement required to accept the refined depth
    int nearest_strong_radius = 100;       // search radius (pixels) for the nearest STRONG pixel
    int anchor_ransac_iters = 50;          // RANSAC iterations when choosing anchors
    int anchor_min_inliers = 6;            // minimum inliers of the anchor plane
    int fit_ransac_iters = 50;             // RANSAC iterations of the per-iteration plane fit
    bool median_filter = true;             // run the checkerboard median depth filter on STRONG pixels
    long long rng_seed = -1;               // <0: seed from the clock (upstream); >=0: reproducible seed
};

// Everything that can be changed at run time, gathered for the option registry (apd_config.cpp).
struct RuntimeConfig {
    FusionParams fusion;
    WeakFilterParams weak_filter;
    PipelineParams pipeline;
    PatchMatchParams pm;
};

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
    path dense_folder;
    path result_folder;
    int scale_size = 1;
    float range_scale_min = 0.6f;  // depth search range = cam file range scaled by these two factors
    float range_scale_max = 1.2f;
    float range_pad = 0.0f;        // world units added to both ends of the scaled range
    PatchMatchParams params;
    bool show_medium_result = false;
    bool export_anchor = false;
    bool export_reliable_curve = false;
    int iteration;
    std::string img_ext;
    int used_time;
};

#endif // !_MAIN_H_
