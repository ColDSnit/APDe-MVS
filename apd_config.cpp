/*
Purpose: Implementation of the APD option registry: registration of all run-time options, parsing of
         values given on the command line or in an INI file, and the INI / JSON dumps.
Status:  Testing
Future:  None
*/
#include "apd_config.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>

namespace {

// Formats a number with enough digits to read back the identical binary value.
std::string FormatNumber(double value, int digits) {
    char buffer[64];                                          // large enough for any %.17g output
    std::snprintf(buffer, sizeof(buffer), "%.*g", digits, value);
    return std::string(buffer);
}

// Shortest decimal text that reads back as exactly the same float (0.01f prints as "0.01", not "0.00999999978").
std::string ShortestFloat(float value) {
    for (int digits = 1; digits < 9; ++digits) {
        const std::string text = FormatNumber(value, digits);
        // plain notation is preferred, so "2e+02" keeps growing until it becomes "200"
        if (std::strtof(text.c_str(), nullptr) == value && text.find('e') == std::string::npos) {
            return text;
        }
    }
    return FormatNumber(value, 9);                            // 9 significant digits always round-trip a float
}

// Shortest decimal text that reads back as exactly the same double.
std::string ShortestDouble(double value) {
    for (int digits = 1; digits < 17; ++digits) {
        const std::string text = FormatNumber(value, digits);
        if (std::strtod(text.c_str(), nullptr) == value && text.find('e') == std::string::npos) {
            return text;
        }
    }
    return FormatNumber(value, 17);                           // 17 significant digits always round-trip a double
}

// Accepts the usual spellings of a boolean.
bool ParseBool(const std::string &text, bool &value) {
    std::string lower = text;                                 // copy so the caller's text is untouched
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "true" || lower == "1" || lower == "on" || lower == "yes") {
        value = true;
        return true;
    }
    if (lower == "false" || lower == "0" || lower == "off" || lower == "no") {
        value = false;
        return true;
    }
    return false;
}

// Escapes a string for use inside JSON double quotes.
std::string JsonEscape(const std::string &text) {
    std::string out;
    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == '"' || ch == '\\') {
            out += '\\';                                      // quote and backslash need a backslash
            out += ch;
        } else if (ch == '\n') {
            out += "\\n";
        } else {
            out += ch;
        }
    }
    return out;
}

// "fusion.depth_mode" -> "fusion"
std::string GroupOf(const std::string &name) {
    return name.substr(0, name.find('.'));
}

// "fusion.depth_mode" -> "depth_mode"
std::string KeyOf(const std::string &name) {
    return name.substr(name.find('.') + 1);
}

}  // namespace

void OptionRegistry::Add(const std::string &name, OptionType type, void *target, bool has_range,
                         double min_value, double max_value, const std::string &stage,
                         const std::string &unit, const std::string &help,
                         const std::vector<std::string> &choices) {
    OptionSpec spec;
    spec.name = name;
    spec.type = type;
    spec.target = target;
    spec.has_range = has_range;
    spec.min_value = min_value;
    spec.max_value = max_value;
    spec.choices = choices;
    spec.stage = stage;
    spec.unit = unit;
    spec.help = help;
    spec.default_text = ValueText(spec);                      // the struct still holds its default here
    specs.push_back(spec);
}

OptionRegistry::OptionRegistry(RuntimeConfig &cfg) : config(cfg) {
    const char *PM = "patchmatch";                            // changing it needs a full APD run
    const char *FU = "fusion";                                // changing it only needs --only_fuse true
    FusionParams &f = cfg.fusion;
    WeakFilterParams &w = cfg.weak_filter;
    PipelineParams &p = cfg.pipeline;
    PatchMatchParams &m = cfg.pm;
    const double BIG = 1e9;                                   // "no practical upper limit"

    std::vector<std::string> depth_modes;                     // index must equal DepthTestMode
    depth_modes.push_back("relative");
    depth_modes.push_back("absolute");
    depth_modes.push_back("pixel");
    depth_modes.push_back("off");
    std::vector<std::string> weak_depth_modes;                // the weak filter only knows these two
    weak_depth_modes.push_back("relative");
    weak_depth_modes.push_back("absolute");

    std::vector<std::string> camera_models;                   // index must equal CameraModel
    camera_models.push_back("pinhole");
    camera_models.push_back("orthographic");

    // ---------------------------------------------------------------- camera model
    Add("camera.model", OPTION_ENUM, &p.camera_model, false, 0, 0, PM, "",
        "Projection model of all cameras: pinhole, or orthographic for telecentric lenses written as "
        "long-focal-length pinhole equivalents (magnification = focal length / reference depth)",
        camera_models);
    Add("camera.ortho_ref_depth", OPTION_FLOAT, &p.ortho_ref_depth, true, 0, BIG, PM, "world",
        "Orthographic reference depth at which the pinhole-equivalent focal length is exact "
        "(0: per camera, the middle of its depth search range)");

    // ---------------------------------------------------------------- fusion
    Add("fusion.depth_mode", OPTION_ENUM, &f.depth_mode, false, 0, 0, FU, "",
        "Depth agreement test: relative (|dd|/d), absolute (world units), pixel (shift in the source image), off",
        depth_modes);
    Add("fusion.depth_rel", OPTION_FLOAT, &f.depth_rel, true, 0, BIG, FU, "ratio",
        "Depth tolerance in relative mode");
    Add("fusion.depth_abs", OPTION_FLOAT, &f.depth_abs, true, 0, BIG, FU, "world",
        "Depth tolerance in absolute mode, in the units of the camera translations");
    Add("fusion.depth_px", OPTION_FLOAT, &f.depth_px, true, 0, BIG, FU, "px",
        "Depth tolerance in pixel mode: allowed shift in the source image caused by the depth difference");
    Add("fusion.reproj_px", OPTION_FLOAT, &f.reproj_px, true, 0, BIG, FU, "px",
        "Forward-backward reprojection tolerance in the reference image");
    Add("fusion.normal_test", OPTION_BOOL, &f.normal_test, false, 0, 0, FU, "",
        "Use normals in the consistency gate and score (normals are compared in the world frame)");
    Add("fusion.normal_max_rad", OPTION_FLOAT, &f.normal_max_rad, true, 0, 3.2, FU, "rad",
        "Largest normal angle between reference and source (0.174533 rad = 10 deg)");
    Add("fusion.score_w_reproj", OPTION_FLOAT, &f.score_w_reproj, true, 0, BIG, FU, "1/px",
        "Weight of the reprojection error in the consistency score exp(-sum)");
    Add("fusion.score_w_depth", OPTION_FLOAT, &f.score_w_depth, true, 0, BIG, FU, "",
        "Weight of the relative depth error in the score (relative mode)");
    Add("fusion.score_w_depth_scaled", OPTION_FLOAT, &f.score_w_depth_scaled, true, 0, BIG, FU, "",
        "Weight of depth error divided by its tolerance in the score (absolute and pixel modes)");
    Add("fusion.score_w_normal", OPTION_FLOAT, &f.score_w_normal, true, 0, BIG, FU, "1/rad",
        "Weight of the normal angle in the score");
    Add("fusion.score_thresh_strong", OPTION_FLOAT, &f.score_thresh_strong, true, 0, 1, FU, "",
        "Mean score a textured (STRONG) pixel must exceed to be fused");
    Add("fusion.score_thresh_weak", OPTION_FLOAT, &f.score_thresh_weak, true, 0, 1, FU, "",
        "Mean score a low-texture (WEAK) pixel must exceed to be fused");
    Add("fusion.min_consistent", OPTION_INT, &f.min_consistent, true, 1, MAX_IMAGES, FU, "views",
        "Minimum number of agreeing source views");
    Add("fusion.mask_used_pixels", OPTION_BOOL, &f.mask_used_pixels, false, 0, 0, FU, "",
        "Mark agreeing source pixels as consumed so they do not create duplicate points");
    Add("fusion.average_position", OPTION_BOOL, &f.average_position, false, 0, 0, FU, "",
        "Fused point = mean of the reference point and the agreeing source points (false: reference point)");
    Add("fusion.view_min_angle_deg", OPTION_FLOAT, &f.view_min_angle_deg, true, -1, 180, FU, "deg",
        "Skip source views whose optical axis is closer than this to the reference axis (<0: off)");
    Add("fusion.view_max_angle_deg", OPTION_FLOAT, &f.view_max_angle_deg, true, -1, 180, FU, "deg",
        "Skip source views whose optical axis is further than this from the reference axis (<0: off)");
    Add("fusion.incident_max_deg", OPTION_FLOAT, &f.incident_max_deg, true, -1, 180, FU, "deg",
        "A pixel takes part in fusion (as reference or source) only if the angle between its normal and the direction "
        "to its own camera is at most this (<0: off)");
    Add("fusion.incident_sigma_deg", OPTION_FLOAT, &f.incident_sigma_deg, true, -1, 180, FU, "deg",
        "Soft incident-angle prior exp(-k^2/2s^2) on every source term and on the reference score (<0: off; "
        "Schoenberger et al. 2016 use 45)");
    Add("fusion.silhouette_trim_px", OPTION_INT, &f.silhouette_trim_px, true, 0, 100000, FU, "px",
        "Pixels within this many depth-map pixels of the view's silhouette edge (scene/sa_masks) take no part in "
        "fusion (0: off)");

    // ---------------------------------------------------------------- weak visibility filter
    Add("weakfilter.max_view_angle_deg", OPTION_FLOAT, &w.max_view_angle_deg, true, 0, 180, FU, "deg",
        "Source views separated by more than this angle at the point are ignored");
    Add("weakfilter.depth_mode", OPTION_ENUM, &w.depth_mode, false, 0, 0, FU, "",
        "Occlusion margin type: relative or absolute", weak_depth_modes);
    Add("weakfilter.depth_rel", OPTION_FLOAT, &w.depth_rel, true, 0, BIG, FU, "ratio",
        "Occlusion margin relative to the source depth");
    Add("weakfilter.depth_abs", OPTION_FLOAT, &w.depth_abs, true, 0, BIG, FU, "world",
        "Occlusion margin in world units");
    Add("weakfilter.strong_occluded_min", OPTION_INT, &w.strong_occluded_min, true, 1, MAX_IMAGES, FU, "views",
        "Number of STRONG source pixels lying behind a WEAK point that removes it");
    Add("weakfilter.weak_occluded_min", OPTION_INT, &w.weak_occluded_min, true, 1, MAX_IMAGES, FU, "views",
        "Number of lower-confidence WEAK source pixels lying behind a WEAK point that removes it");
    Add("weakfilter.confidence_as_uchar", OPTION_BOOL, &w.confidence_as_uchar, false, 0, 0, FU, "",
        "Read the confidence map with its stored type (false keeps the upstream float read)");

    // ---------------------------------------------------------------- depth range
    Add("depth.range_scale_min", OPTION_FLOAT, &p.range_scale_min, true, 0, BIG, PM, "ratio",
        "Search range lower bound = DEPTH_MIN of the cam file times this (1 = no widening)");
    Add("depth.range_scale_max", OPTION_FLOAT, &p.range_scale_max, true, 0, BIG, PM, "ratio",
        "Search range upper bound = DEPTH_MAX of the cam file times this (1 = no widening)");

    Add("depth.range_pad", OPTION_FLOAT, &p.range_pad, true, -BIG, BIG, PM, "world",
        "World units added to both ends of the scaled search range (negative shrinks it)");

    // ---------------------------------------------------------------- view selection from pair.txt
    Add("views.all_pairs", OPTION_BOOL, &p.all_pairs, false, 0, 0, PM, "",
        "Use every other image as source view; pair.txt only sets the order (scores are ignored)");
    Add("views.skip_self", OPTION_BOOL, &p.skip_self, false, 0, 0, PM, "",
        "Ignore pair.txt entries that name the reference image itself");
    Add("views.min_pair_score", OPTION_FLOAT, &p.min_pair_score, true, -BIG, BIG, PM, "",
        "pair.txt source views with a score at or below this are skipped");
    Add("views.max_src", OPTION_INT, &p.max_src_views, true, -1, MAX_IMAGES - 1, PM, "views",
        "Keep only the first N source views of every reference (<=0: all)");
    Add("views.min_angle_deg", OPTION_FLOAT, &p.view_min_angle_deg, true, -1, 180, PM, "deg",
        "Drop source views whose optical axis is closer than this to the reference axis (<0: off)");
    Add("views.max_angle_deg", OPTION_FLOAT, &p.view_max_angle_deg, true, -1, 180, PM, "deg",
        "Drop source views whose optical axis is further than this from the reference axis (<0: off)");

    // ---------------------------------------------------------------- coarse-to-fine schedule
    Add("pipeline.round_max_size", OPTION_INT, &p.round_max_size, true, 16, 1000000, PM, "px",
        "The image is halved until its longer side is at most this; sets the number of scales");
    Add("pipeline.rounds", OPTION_INT, &p.rounds, true, -1, 16, PM, "",
        "Force the number of scales (<=0: derive from round_max_size)");
    Add("pipeline.extra_rounds", OPTION_INT, &p.extra_rounds, true, 0, 8, PM, "",
        "Scales added to the automatic count (each one starts at half the size of the previous coarsest; "
        "ignored when rounds > 0; never shrinks the image below 32 px)");
    Add("pipeline.geom_iterations", OPTION_INT, &p.geom_iterations, true, 0, 64, PM, "",
        "Geometric-consistency passes per scale");
    Add("pipeline.max_iterations", OPTION_INT, &p.max_iterations, true, 1, 64, PM, "",
        "Checkerboard propagation iterations per pass");
    Add("pipeline.init_weak_peak_radius", OPTION_INT, &p.init_weak_peak_radius, true, 0, 30, PM, "steps",
        "Cost-curve peak tolerance of the first pass of each scale");
    Add("pipeline.weak_peak_start", OPTION_INT, &p.weak_peak_start, true, 0, 30, PM, "steps",
        "Peak tolerance of geometric pass j = max(start - step * j, min)");
    Add("pipeline.weak_peak_step", OPTION_INT, &p.weak_peak_step, true, 0, 30, PM, "steps",
        "See weak_peak_start");
    Add("pipeline.weak_peak_min", OPTION_INT, &p.weak_peak_min, true, 0, 30, PM, "steps",
        "See weak_peak_start");
    Add("pipeline.ransac_base", OPTION_DOUBLE, &p.ransac_base, true, 0, BIG, PM, "ratio",
        "Anchor plane inlier threshold = base - scale_index * step (fraction of the depth range)");
    Add("pipeline.ransac_step", OPTION_DOUBLE, &p.ransac_step, true, 0, BIG, PM, "ratio",
        "See ransac_base");
    Add("pipeline.rotate_time_max", OPTION_INT, &p.rotate_time_max, true, 1, 4, PM, "",
        "Anchor search rotations per direction = min(2^scale_index, this)");

    // ---------------------------------------------------------------- PatchMatch core
    Add("pm.strong_radius", OPTION_INT, &m.strong_radius, true, 1, 64, PM, "px",
        "NCC window radius of textured pixels and of the centre patch");
    Add("pm.strong_increment", OPTION_INT, &m.strong_increment, true, 1, 64, PM, "px",
        "Sampling step inside the strong NCC window");
    Add("pm.weak_radius", OPTION_INT, &m.weak_radius, true, 1, 64, PM, "px",
        "NCC window radius around each anchor of a low-texture pixel");
    Add("pm.weak_increment", OPTION_INT, &m.weak_increment, true, 1, 64, PM, "px",
        "Sampling step inside the anchor NCC window");
    Add("pm.top_k", OPTION_INT, &m.top_k, true, 1, MAX_IMAGES - 1, PM, "views",
        "Number of best source views used for the initial cost and view selection");
    Add("pm.geom_factor", OPTION_FLOAT, &p.geom_factor, true, -1, BIG, PM, "1/px",
        "Weight of the geometric consistency cost (<0: dataset default, 0.2 or 0.05 for TaT)");
    Add("pm.geom_max_cost", OPTION_FLOAT, &m.geom_max_cost, true, 0, BIG, PM, "px",
        "Cap of the geometric consistency (forward-backward reprojection) cost");
    Add("pm.depth_perturbation", OPTION_FLOAT, &m.depth_perturbation, true, 0, 1, PM, "ratio",
        "Relative depth perturbation of the local refinement hypotheses");
    Add("pm.normal_perturbation", OPTION_FLOAT, &m.normal_perturbation, true, 0, 1, PM, "pi",
        "Normal perturbation of the local refinement hypotheses, as a fraction of pi");
    Add("pm.refine_init_margin", OPTION_DOUBLE, &m.refine_init_margin, true, 0, 2, PM, "cost",
        "Cost improvement needed to replace an upsampled hypothesis in the first pass of a scale");
    Add("pm.weak_center_weight", OPTION_DOUBLE, &m.weak_center_weight, true, 0, 1, PM, "",
        "Deformable NCC = w * centre patch + (1 - w) * anchor patches");
    Add("pm.median_filter", OPTION_BOOL, &m.median_filter, false, 0, 0, PM, "",
        "Median-filter the depth of textured pixels after propagation");
    Add("pm.local_refine", OPTION_BOOL, &m.local_refine, false, 0, 0, PM, "",
        "Final per-pixel depth refinement over disparity steps");
    Add("pm.local_refine_radius", OPTION_INT, &m.local_refine_radius, true, 0, 64, PM, "steps",
        "Search radius of the final refinement");
    Add("pm.local_refine_margin", OPTION_DOUBLE, &m.local_refine_margin, true, 0, 2, PM, "cost",
        "Cost improvement needed to accept the refined depth");
    Add("pm.rng_seed", OPTION_INT64, &m.rng_seed, true, -1, 9e18, PM, "",
        "Random seed (<0: seed from the clock, runs are not repeatable)");

    // ---------------------------------------------------------------- per-pixel view selection
    Add("viewsel.prior_selected", OPTION_FLOAT, &m.vs_prior_selected, true, 0, 1, PM, "",
        "Prior of a view that a neighbouring pixel selected");
    Add("viewsel.prior_unselected", OPTION_FLOAT, &m.vs_prior_unselected, true, 0, 1, PM, "",
        "Prior of a view that a neighbouring pixel did not select");
    Add("viewsel.cost_thresh_init", OPTION_DOUBLE, &m.vs_cost_thresh_init, true, 0, 2, PM, "cost",
        "Good-cost threshold = init * exp(-iteration^2 / decay)");
    Add("viewsel.cost_thresh_decay", OPTION_FLOAT, &m.vs_cost_thresh_decay, true, 1e-6, BIG, PM, "",
        "See cost_thresh_init");
    Add("viewsel.good_sigma", OPTION_FLOAT, &m.vs_good_sigma, true, 1e-6, BIG, PM, "",
        "View weight of a good cost = exp(-cost^2 / sigma)");
    Add("viewsel.fallback_sigma", OPTION_FLOAT, &m.vs_fallback_sigma, true, 1e-6, BIG, PM, "",
        "View weight when too few costs are good = exp(-threshold^2 / sigma)");
    Add("viewsel.bad_cost", OPTION_FLOAT, &m.vs_bad_cost, true, 0, 2, PM, "cost",
        "NCC cost above which a hypothesis counts as bad for a view");
    Add("viewsel.min_good", OPTION_INT, &m.vs_min_good, true, 0, 8, PM, "",
        "A view needs more than this many good hypotheses (of 8)");
    Add("viewsel.max_bad", OPTION_INT, &m.vs_max_bad, true, 0, 9, PM, "",
        "A view is rejected at this many bad hypotheses (of 8)");
    Add("viewsel.num_samples", OPTION_INT, &m.vs_num_samples, true, 1, 255, PM, "",
        "Monte-Carlo view samples drawn per pixel");

    // ---------------------------------------------------------------- weak / strong classification, anchors
    Add("weak.border_margin", OPTION_INT, &m.border_margin, true, 0, 1024, PM, "px",
        "Image border that is never classified and never used as anchor");
    Add("weak.max_peak_cost", OPTION_FLOAT, &m.weak_max_peak_cost, true, 0, 2, PM, "cost",
        "A cost-curve minimum above this makes the pixel WEAK");
    Add("weak.strong_single_peak_cost", OPTION_FLOAT, &m.strong_single_peak_cost, true, 0, 2, PM, "cost",
        "A single-minimum cost curve below this makes the pixel STRONG");
    Add("weak.strong_multi_peak_var", OPTION_FLOAT, &m.strong_multi_peak_var, true, 0, 2, PM, "cost",
        "A multi-minimum cost curve with a spread above this makes the pixel STRONG");
    Add("weak.filter_radius", OPTION_INT, &m.weak_filter_radius, true, 0, 64, PM, "px",
        "A STRONG pixel without another STRONG pixel in this radius becomes UNKNOWN");
    Add("weak.nearest_strong_radius", OPTION_INT, &m.nearest_strong_radius, true, 1, 1024, PM, "px",
        "Search radius for the nearest STRONG pixel (cost grows with the square)");
    Add("weak.anchor_ransac_iters", OPTION_INT, &m.anchor_ransac_iters, true, 1, 100000, PM, "",
        "RANSAC iterations when choosing the anchors of a WEAK pixel");
    Add("weak.anchor_min_inliers", OPTION_INT, &m.anchor_min_inliers, true, 3, 32, PM, "",
        "Minimum inliers of the anchor plane");
    Add("weak.fit_ransac_iters", OPTION_INT, &m.fit_ransac_iters, true, 1, 100000, PM, "",
        "RANSAC iterations of the per-iteration plane fit of a WEAK pixel");

    // ---------------------------------------------------------------- confidence map
    Add("conf.reproj_px", OPTION_FLOAT, &m.conf_reproj_px, true, 0, BIG, PM, "px",
        "Reprojection tolerance of the confidence vote");
    Add("conf.depth_rel", OPTION_FLOAT, &m.conf_depth_rel, true, 0, BIG, PM, "ratio",
        "Relative depth tolerance of the confidence vote");
    Add("conf.w_exist", OPTION_INT, &m.conf_w_exist, true, 0, 8, PM, "votes",
        "Votes for a source view that has a depth at the projected pixel");
    Add("conf.w_reproj", OPTION_INT, &m.conf_w_reproj, true, 0, 8, PM, "votes",
        "Votes for passing the reprojection test");
    Add("conf.w_depth", OPTION_INT, &m.conf_w_depth, true, 0, 8, PM, "votes",
        "Votes for passing the depth test");
}

void OptionRegistry::AddToDescription(boost::program_options::options_description &desc) const {
    namespace opt = boost::program_options;
    for (size_t i = 0; i < specs.size(); ++i) {
        const OptionSpec &spec = specs[i];
        const std::string text = spec.help + " [default " + spec.default_text + "]";
        // Every option is read as text and converted by SetFromText, so defaults never pass through a parser.
        desc.add_options()(spec.name.c_str(), opt::value<std::string>(), text.c_str());
    }
}

std::string OptionRegistry::ValueText(const OptionSpec &spec) const {
    switch (spec.type) {
        case OPTION_INT:
            return std::to_string(*static_cast<int *>(spec.target));
        case OPTION_INT64:
            return std::to_string(*static_cast<long long *>(spec.target));
        case OPTION_FLOAT:
            return ShortestFloat(*static_cast<float *>(spec.target));
        case OPTION_DOUBLE:
            return ShortestDouble(*static_cast<double *>(spec.target));
        case OPTION_BOOL:
            return *static_cast<bool *>(spec.target) ? "true" : "false";
        case OPTION_ENUM: {
            const int index = *static_cast<int *>(spec.target);
            if (index >= 0 && index < static_cast<int>(spec.choices.size())) {
                return spec.choices[index];
            }
            return "?";
        }
    }
    return "";
}

bool OptionRegistry::SetFromText(const OptionSpec &spec, const std::string &text, std::string &error) const {
    const char *begin = text.c_str();
    char *end = nullptr;
    errno = 0;
    double as_double = 0.0;                                   // value used for the range check
    switch (spec.type) {
        case OPTION_BOOL: {
            bool value = false;
            if (!ParseBool(text, value)) {
                error = "option " + spec.name + ": '" + text + "' is not a boolean";
                return false;
            }
            *static_cast<bool *>(spec.target) = value;
            return true;
        }
        case OPTION_ENUM: {
            for (size_t i = 0; i < spec.choices.size(); ++i) {
                if (spec.choices[i] == text) {
                    *static_cast<int *>(spec.target) = static_cast<int>(i);
                    return true;
                }
            }
            error = "option " + spec.name + ": '" + text + "' is not one of the allowed choices";
            return false;
        }
        case OPTION_INT:
        case OPTION_INT64: {
            const long long value = std::strtoll(begin, &end, 10);
            if (end == begin || *end != '\0' || errno != 0) {
                error = "option " + spec.name + ": '" + text + "' is not an integer";
                return false;
            }
            as_double = static_cast<double>(value);
            if (spec.has_range && (as_double < spec.min_value || as_double > spec.max_value)) {
                break;                                        // reported below
            }
            if (spec.type == OPTION_INT) {
                *static_cast<int *>(spec.target) = static_cast<int>(value);
            } else {
                *static_cast<long long *>(spec.target) = value;
            }
            return true;
        }
        case OPTION_FLOAT: {
            const float value = std::strtof(begin, &end);     // direct float parse avoids double rounding
            if (end == begin || *end != '\0' || errno != 0) {
                error = "option " + spec.name + ": '" + text + "' is not a number";
                return false;
            }
            as_double = value;
            if (spec.has_range && !(as_double >= spec.min_value && as_double <= spec.max_value)) {
                break;
            }
            *static_cast<float *>(spec.target) = value;
            return true;
        }
        case OPTION_DOUBLE: {
            const double value = std::strtod(begin, &end);
            if (end == begin || *end != '\0' || errno != 0) {
                error = "option " + spec.name + ": '" + text + "' is not a number";
                return false;
            }
            as_double = value;
            if (spec.has_range && !(as_double >= spec.min_value && as_double <= spec.max_value)) {
                break;
            }
            *static_cast<double *>(spec.target) = value;
            return true;
        }
    }
    error = "option " + spec.name + ": " + text + " is outside [" + FormatNumber(spec.min_value, 9) + ", " +
            FormatNumber(spec.max_value, 9) + "]";
    return false;
}

bool OptionRegistry::Apply(const boost::program_options::variables_map &vm, std::string &error) {
    for (size_t i = 0; i < specs.size(); ++i) {
        const OptionSpec &spec = specs[i];
        if (vm.count(spec.name) == 0) {
            continue;                                         // not given: the default stays untouched
        }
        if (!SetFromText(spec, vm[spec.name].as<std::string>(), error)) {
            return false;
        }
    }
    return Validate(error);
}

bool OptionRegistry::Validate(std::string &error) const {
    const PatchMatchParams &m = config.pm;
    // The kernels allocate the cost curve and the anchor tables with fixed sizes; keep the options inside them.
    if (m.strong_increment > 2 * m.strong_radius || m.weak_increment > 2 * m.weak_radius) {
        error = "pm.*_increment must not exceed twice the matching radius (the window would hold one sample)";
        return false;
    }
    if (config.pipeline.weak_peak_min > config.pipeline.weak_peak_start) {
        error = "pipeline.weak_peak_min must not exceed pipeline.weak_peak_start";
        return false;
    }
    return true;
}

std::string OptionRegistry::ToIni() const {
    std::string out = "# APD effective configuration; pass this file back with --config\n";
    std::string group;
    for (size_t i = 0; i < specs.size(); ++i) {
        const OptionSpec &spec = specs[i];
        if (GroupOf(spec.name) != group) {
            group = GroupOf(spec.name);
            out += "\n[" + group + "]\n";                     // boost reads [group] + key as "group.key"
        }
        out += KeyOf(spec.name) + " = " + ValueText(spec) + "\n";
    }
    return out;
}

std::string OptionRegistry::ToJson() const {
    std::string out = "{\n  \"schema_version\": 1,\n  \"program\": \"APD\",\n  \"options\": [\n";
    for (size_t i = 0; i < specs.size(); ++i) {
        const OptionSpec &spec = specs[i];
        std::string type_name;
        std::string default_json = spec.default_text;         // numbers and booleans are valid JSON as they are
        switch (spec.type) {
            case OPTION_INT:
            case OPTION_INT64:
                type_name = "int";
                break;
            case OPTION_FLOAT:
            case OPTION_DOUBLE:
                type_name = "float";
                break;
            case OPTION_BOOL:
                type_name = "bool";
                break;
            case OPTION_ENUM:
                type_name = "enum";
                default_json = "\"" + JsonEscape(spec.default_text) + "\"";
                break;
        }
        out += "    {\"name\": \"" + spec.name + "\", \"group\": \"" + GroupOf(spec.name) + "\", \"key\": \"" +
               KeyOf(spec.name) + "\", \"type\": \"" + type_name + "\", \"default\": " + default_json;
        if (spec.has_range) {
            out += ", \"min\": " + ShortestDouble(spec.min_value) + ", \"max\": " + ShortestDouble(spec.max_value);
        }
        if (!spec.choices.empty()) {
            out += ", \"choices\": [";
            for (size_t c = 0; c < spec.choices.size(); ++c) {
                out += (c == 0 ? "\"" : ", \"") + spec.choices[c] + "\"";
            }
            out += "]";
        }
        out += ", \"unit\": \"" + JsonEscape(spec.unit) + "\", \"stage\": \"" + spec.stage + "\", \"help\": \"" +
               JsonEscape(spec.help) + "\"}";
        out += (i + 1 < specs.size() ? ",\n" : "\n");
    }
    out += "  ]\n}\n";
    return out;
}
