/*
Purpose: Plain-data option structs for the host-side (CPU) stages of APD: fusion, the weak-pixel
         visibility filter and the multi-scale pipeline schedule. Every default equals the literal
         that upstream APDe-MVS hard-coded, so an unconfigured run behaves like upstream.
Status:  Testing
Future:  Orthographic (telecentric) camera model switch; adaptive matching windows.
*/
#ifndef _APD_OPTIONS_H_
#define _APD_OPTIONS_H_

// How the depth agreement between a reference pixel and a source view is measured.
enum DepthTestMode {
    DEPTH_TEST_RELATIVE = 0,  // |d_src - d_ref| / d_ref            (upstream behaviour)
    DEPTH_TEST_ABSOLUTE = 1,  // |d_src - d_ref| in world units
    DEPTH_TEST_PIXEL = 2,     // displacement in the source image caused by the depth difference
    DEPTH_TEST_OFF = 3        // no depth test; only reprojection (and normals) gate the match
};

// Options of RunFusion (the general, non Tanks-and-Temples fusion).
struct FusionParams {
    int depth_mode = DEPTH_TEST_RELATIVE;  // one of DepthTestMode
    float depth_rel = 0.01f;               // tolerance for DEPTH_TEST_RELATIVE
    float depth_abs = 0.1f;                // tolerance for DEPTH_TEST_ABSOLUTE, world units
    float depth_px = 1.0f;                 // tolerance for DEPTH_TEST_PIXEL, source-image pixels
    float reproj_px = 2.0f;                // forward-backward reprojection tolerance, reference pixels
    bool normal_test = true;               // false removes normals from both the gate and the score
    float normal_max_rad = 0.174533f;      // normal angle tolerance in radians (upstream literal, 10 degrees)
    float score_w_reproj = 1.0f;           // score = exp(-(w_reproj*reproj + w_depth*depth + w_normal*angle))
    float score_w_depth = 200.0f;          // depth weight in relative mode (multiplies the relative error)
    float score_w_depth_scaled = 2.0f;     // depth weight in absolute/pixel mode (multiplies error/tolerance)
    float score_w_normal = 10.0f;          // normal weight (multiplies the angle in radians)
    float score_thresh_strong = 0.3f;      // mean score a STRONG (textured) pixel must exceed
    float score_thresh_weak = 0.45f;       // mean score a WEAK (low-texture) pixel must exceed
    int min_consistent = 1;                // minimum number of agreeing source views
    bool mask_used_pixels = true;          // mark agreeing source pixels as consumed (no duplicate points)
    bool average_position = false;         // true: fused point = mean of the agreeing 3D points
    float view_min_angle_deg = -1.0f;      // <0 disables; else skip source views whose optical axis is closer
    float view_max_angle_deg = -1.0f;      // <0 disables; else skip source views whose optical axis is further
    // Grazing-view guards (feat/fusion-grazing-guard; all off by default = upstream behaviour).
    float incident_max_deg = -1.0f;        // <0 disables; else a pixel takes part (as reference or as source) only if the
                                           // angle between its normal and the direction to its own camera is <= this
    float incident_sigma_deg = -1.0f;      // <0 disables; else soft incident prior exp(-k^2 / 2 sigma^2) (Schoenberger et
                                           // al. 2016, sigma 45 deg): weights each source term and the reference score
    int silhouette_trim_px = 0;            // 0 disables; else pixels within this many depth-map pixels of the view's own
                                           // silhouette edge (scene/sa_masks, nonzero = object) take no part in fusion
};

// Options of WeakVisFilter (drops WEAK pixels that other views see through).
struct WeakFilterParams {
    float max_view_angle_deg = 80.0f;      // views further apart than this (at the point) are ignored
    int depth_mode = DEPTH_TEST_RELATIVE;  // relative or absolute only
    float depth_rel = 0.01f;               // occlusion margin, relative to the source depth
    float depth_abs = 0.1f;                // occlusion margin in world units
    int strong_occluded_min = 2;           // STRONG source pixels behind the point needed to drop it
    int weak_occluded_min = 4;             // WEAK source pixels behind the point needed to drop it
    bool confidence_as_uchar = false;      // false keeps the upstream read of the uchar map as float
};

// Options of the coarse-to-fine schedule in main.cpp and of scene loading.
struct PipelineParams {
    int round_max_size = 800;              // halve the image until its longer side is <= this
    int rounds = -1;                       // >0 forces the number of scales
    int extra_rounds = 0;                  // added to the automatic number of scales (ignored when rounds > 0)
    int geom_iterations = 3;               // geometric-consistency passes per scale
    int max_iterations = 3;                // checkerboard propagation iterations per pass
    int init_weak_peak_radius = 6;         // weak_peak_radius of the first pass of each scale
    int weak_peak_start = 4;               // weak_peak_radius = max(start - step * j, min) in geometric pass j
    int weak_peak_step = 2;
    int weak_peak_min = 2;
    double ransac_base = 0.01;             // ransac_threshold = base - scale_index * step
    double ransac_step = 0.00125;
    int rotate_time_max = 4;               // anchor search rotations = min(2^scale_index, this); at most 4
    float geom_factor = -1.0f;             // <0 keeps the dataset default (0.2, or 0.05 for Tanks and Temples)
    float range_scale_min = 0.6f;          // depth search range = [file_min * this, file_max * range_scale_max]
    float range_scale_max = 1.2f;
    int camera_model = 0;                  // 0 pinhole (upstream), 1 orthographic (telecentric lenses)
    float ortho_ref_depth = 0.0f;          // <=0: per camera, the middle of its depth range; >0: this depth for all
    float range_pad = 0.0f;                // world units added to both ends of the scaled range
    bool all_pairs = false;                // true: every other image is a source view, pair.txt only orders them
    bool skip_self = false;                // true: a pair.txt entry naming the reference itself is ignored
    float min_pair_score = 0.0f;           // pair.txt entries with score <= this are skipped
    int max_src_views = -1;                // >0 keeps only the first N source views of each reference
    float view_min_angle_deg = -1.0f;      // <0 disables; else drop source views by optical-axis angle
    float view_max_angle_deg = -1.0f;
};

#endif // !_APD_OPTIONS_H_
