#include "main.h"
#include "APD.h"
#include "apd_config.h"
#include <algorithm>

using namespace boost::filesystem;
namespace opt = boost::program_options;

// Parses the command line and the optional --config INI file; run-time options land in registry's config.
opt::variables_map ParseArgs(int argc, char **argv, OptionRegistry &registry) {
    opt::options_description desc("Allowed options");
    opt::options_description tuning("Run-time options (also accepted in the --config INI file as [group] key = value)");
    registry.AddToDescription(tuning);
    desc.add_options()
            ("dense_folder,d", opt::value<std::string>()->required(), "path to dense folder")
            ("gpu_index,g", opt::value<int>()->default_value(0), "gpu index")
            ("dataset,D", opt::value<std::string>()->default_value(std::string("DTU")),
             "dataset name, DTU, ETH3D or Tanks and Temples")
            ("only_fuse,f", opt::value<bool>()->default_value(false), "only fuse depths")
            ("no_fuse,F", opt::value<bool>()->default_value(false), "skip fuse")
            ("memory_cache,m", opt::value<bool>()->default_value(true), "use memory cache")
            ("use_sa,s", opt::value<bool>()->default_value(true), "use segment anything results")
            ("use_impetus,i", opt::value<bool>()->default_value(true), "use impetus")
            ("weak_filter,w", opt::value<bool>()->default_value(true), "use weak filter")
            ("flush", opt::value<bool>()->default_value(false), "Flush mat to disk")
            ("export_anchor,n", opt::value<bool>()->default_value(false), "Export anchor points to disk")
            ("export_curve,r", opt::value<bool>()->default_value(false), "Export reliable curve to disk")
            ("export_color,c", opt::value<bool>()->default_value(true), "Export ply with color")
            ("config", opt::value<std::string>(), "INI file with run-time options; command-line values win")
            ("dump_options_json", "print the run-time option schema as JSON and exit")
            ("help,h", "produce help message");
    desc.add(tuning);

    opt::variables_map vm;
    try {
        opt::store(opt::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            exit(0);
        }
        if (vm.count("dump_options_json")) {
            std::cout << registry.ToJson();
            exit(0);
        }
        // The APD_CONFIG environment variable stands in for --config, so a caller that cannot pass new
        // arguments (an unmodified wrapper script) can still select a configuration file.
        const char *env_config = std::getenv("APD_CONFIG");
        if (vm.count("config") || (env_config != nullptr && env_config[0] != '\0')) {
            const std::string config_path = vm.count("config") ? vm["config"].as<std::string>()
                                                               : std::string(env_config);
            std::cout << "Run-time options file: " << config_path << std::endl;
            std::ifstream config_stream(config_path.c_str());
            if (!config_stream.is_open()) {
                std::cout << "Error: can not open config file: " << config_path << std::endl;
                exit(-1);
            }
            // Values already stored from the command line are kept, so the command line wins.
            opt::store(opt::parse_config_file(config_stream, tuning), vm);
        }
        opt::notify(vm);
        std::string error;
        if (!registry.Apply(vm, error)) {
            std::cout << "Error: " << error << std::endl;
            exit(-1);
        }
    }
    catch (opt::error &e) {
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << desc << std::endl;
        exit(-1);
    }
    return vm;
}


void GenerateSampleList(const path &dense_folder, std::vector<Problem> &problems, const PipelineParams &pipeline) {
    path cluster_list_path = dense_folder / path("pair.txt");
    path image_folder = dense_folder / path("images");
    path cam_folder = dense_folder / path("cams");
    // The cameras are only read here when the angle window is switched on.
    const bool use_angle_window = (pipeline.view_min_angle_deg >= 0.0f || pipeline.view_max_angle_deg >= 0.0f);
    problems.clear();
    ifstream file(cluster_list_path);
    std::stringstream iss;
    std::string line;
    std::vector<std::string> support_ext = {".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"};

    int num_images;
    iss.clear();
    std::getline(file, line);
    iss.str(line);
    iss >> num_images;

    for (int i = 0; i < num_images; ++i) {
        Problem problem;
        problem.src_image_ids.clear();
        iss.clear();
        std::getline(file, line);
        iss.str(line);
        iss >> problem.ref_image_id;

        problem.dense_folder = dense_folder;
        problem.result_folder = dense_folder / path("APD") / path(ToFormatIndex(problem.ref_image_id));
        create_directory(problem.result_folder);

        int num_src_images;
        iss.clear();
        std::getline(file, line);
        iss.str(line);
        iss >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            iss >> id >> score;
            // all_pairs ignores the score; otherwise upstream's "score <= 0 is skipped" rule applies
            if (!pipeline.all_pairs && score <= pipeline.min_pair_score) {
                continue;
            }
            // pair lists built from sparse covisibility can name the reference itself with score 0
            if ((pipeline.skip_self || pipeline.all_pairs) && id == problem.ref_image_id) {
                continue;
            }
            if (pipeline.all_pairs && std::find(problem.src_image_ids.begin(), problem.src_image_ids.end(), id) !=
                                      problem.src_image_ids.end()) {
                continue;
            }
            problem.src_image_ids.push_back(id);
        }
        // get image's ext
        std::string ext;
        for (auto &support: support_ext) {
            path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + support);
            if (exists(image_path)) {
                ext = support;
                break;
            }
        }
        if (ext.empty()) {
            std::cout << "Error: can not find image: " << ToFormatIndex(problem.ref_image_id) << std::endl;
            exit(-1);
        }
        problem.img_ext = ext;
        problem.used_time = 0;
        problems.push_back(problem);
    }
    // ---- optional post-processing of the source lists; with default options nothing below changes them ----
    for (auto &problem: problems) {
        if (pipeline.all_pairs) {
            // append every other image that pair.txt did not name for this reference
            for (const auto &other: problems) {
                const int id = other.ref_image_id;
                if (id != problem.ref_image_id && std::find(problem.src_image_ids.begin(),
                                                            problem.src_image_ids.end(), id) ==
                                                  problem.src_image_ids.end()) {
                    problem.src_image_ids.push_back(id);
                }
            }
        }
        if (use_angle_window) {
            Camera ref_camera;
            const bool ref_ok = ReadCamera(cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt"),
                                           ref_camera);
            std::vector<int> kept;
            if (!ref_ok) {
                // no geometry to decide on: keep every source and say so once
                std::cout << "WARNING: views.*_angle_deg: cam file of view " << problem.ref_image_id
                          << " cannot be read; its sources are not filtered (PatchMatch stops on that file, "
                          << "fusion excludes the view)" << std::endl;
                kept = problem.src_image_ids;
            } else {
                for (const int id: problem.src_image_ids) {
                    Camera src_camera;
                    if (!ReadCamera(cam_folder / path(ToFormatIndex(id) + "_cam.txt"), src_camera)) {
                        // without geometry the view cannot serve as a source (PatchMatch would stop on the file)
                        std::cout << "WARNING: views.*_angle_deg: cam file of view " << id << " cannot be read; it "
                                  << "is dropped as a source of view " << problem.ref_image_id << std::endl;
                    } else if (ViewAngleAllowed(ref_camera, src_camera, pipeline.view_min_angle_deg,
                                                pipeline.view_max_angle_deg)) {
                        kept.push_back(id);
                    }
                }
            }
            problem.src_image_ids = kept;
        }
        if (pipeline.max_src_views > 0 && (int) problem.src_image_ids.size() > pipeline.max_src_views) {
            problem.src_image_ids.resize(pipeline.max_src_views);  // pair.txt order = best first
        }
        if (pipeline.all_pairs || use_angle_window || pipeline.max_src_views > 0) {
            std::cout << "Source views of image " << problem.ref_image_id << ":";
            for (const int id: problem.src_image_ids) {
                std::cout << " " << id;
            }
            std::cout << std::endl;
        }
    }
}

bool CheckImages(const std::vector<Problem> &problems) {
    if (problems.size() == 0) {
        return false;
    }
    path image_path = problems[0].dense_folder / path("images") /
                      path(ToFormatIndex(problems[0].ref_image_id) + problems[0].img_ext);
    cv::Mat image;
    if (!ReadImage(image_path, image)) {
        return false;
    }
    const int width = image.cols;
    const int height = image.rows;
    for (size_t i = 1; i < problems.size(); ++i) {
        image_path = problems[i].dense_folder / path("images") /
                     path(ToFormatIndex(problems[i].ref_image_id) + problems[i].img_ext);
        if (!ReadImage(image_path, image)) {
            return false;
        }
        if (image.cols != width || image.rows != height) {
            return false;
        }
    }
    return true;
}

int ComputeRoundNum(const std::vector<Problem> &problems, const PipelineParams &pipeline) {
    if (pipeline.rounds > 0) {
        return pipeline.rounds;  // explicit number of scales
    }
    if (problems.size() == 0) {
        return 0;
    }
    path image_path = problems[0].dense_folder / path("images") /
                      path(ToFormatIndex(problems[0].ref_image_id) + problems[0].img_ext);
    cv::Mat image;
    if (!ReadImage(image_path, image)) {
        return 0;
    }
    int max_size = MAX(image.cols, image.rows);
    int round_num = 1;
    while (max_size > pipeline.round_max_size) {  // 800 for TAT & BlendedMVS & DTU and 1000 for ETH3D
        max_size /= 2;
        round_num++;
    }
    // optional extra coarse scales on top of the automatic count; stop before the shorter side drops below 32 px
    int min_size = MIN(image.cols, image.rows) >> (round_num - 1);
    for (int extra = 0; extra < pipeline.extra_rounds && min_size / 2 >= 32; ++extra) {
        min_size /= 2;
        round_num++;
    }
    return round_num;
}

void ProcessProblem(Problem &problem) {
    std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..."
              << std::endl;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    APD APD(problem);
    APD.InuputInitialization();
    APD.CudaSpaceInitialization();
    APD.SetDataPassHelperInCuda();
    std::chrono::steady_clock::time_point start_wo_io = std::chrono::steady_clock::now();
    APD.RunPatchMatch();
    std::chrono::steady_clock::time_point end_wo_io = std::chrono::steady_clock::now();
    printf("RunPatchMatch time: %d ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_wo_io - start_wo_io).count());
    problem.used_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_wo_io - start_wo_io).count();

    int width = APD.GetWidth(), height = APD.GetHeight();
    cv::Mat depth = cv::Mat(height, width, CV_32FC1);
    cv::Mat normal = cv::Mat(height, width, CV_32FC3);
    cv::Mat pixel_states = APD.GetPixelStates();
    cv::Mat confidence = APD.GetConfidence();
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float4 plane_hypothesis = APD.GetPlaneHypothesis(r, c);
            depth.at<float>(r, c) = plane_hypothesis.w;
            // view range, or the pixel's prior band when a depth prior is loaded (identical without one)
            if (!APD.DepthInBounds(r, c, depth.at<float>(r, c))) {
                depth.at<float>(r, c) = 0;
                pixel_states.at<uchar>(r, c) = UNKNOWN;
            }
            normal.at<cv::Vec3f>(r, c) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
        }
    }

    path depth_path = problem.result_folder / path("depths.bin");
    WriteBinMat(depth_path, depth);
    path normal_path = problem.result_folder / path("normals.bin");
    WriteBinMat(normal_path, normal);
    path weak_path = problem.result_folder / path("weak.bin");
    WriteBinMat(weak_path, pixel_states);

    if (problem.params.geom_consistency || problem.params.use_APD) {
        path confidence_path = problem.result_folder / path("confidence.bin");
        WriteBinMat(confidence_path, confidence);
    }

    if (problem.show_medium_result) {
        path depth_img_path = problem.result_folder / path("depth_" + std::to_string(problem.iteration) + ".jpg");
        path normal_img_path = problem.result_folder / path("normal_" + std::to_string(problem.iteration) + ".jpg");
        path weak_img_path = problem.result_folder / path("weak_" + std::to_string(problem.iteration) + ".png");

        ShowDepthMap(depth_img_path, depth, APD.GetDepthMin(), APD.GetDepthMax());
        ShowNormalMap(normal_img_path, normal);
        ShowWeakImage(weak_img_path, pixel_states);
        if (problem.params.geom_consistency || problem.params.use_APD) {
            path confidence_img_path = problem.result_folder / path("confidence_" + std::to_string(problem.iteration) + ".png");
            ShowConfidenceMap(confidence_img_path, confidence);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
    std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

int main(int argc, char **argv) {
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Parse arguments and prepare for processing
    ////////////////////////////////////////////////////////////////////////////////////////////////
    RuntimeConfig config;               // holds every run-time option, initialised to the upstream defaults
    OptionRegistry registry(config);    // describes the options and writes parsed values into config
    opt::variables_map vm = ParseArgs(argc, argv, registry);
    const PipelineParams &pipeline = config.pipeline;
    std::string dense_folder_str = vm["dense_folder"].as<std::string>();
    int gpu_index = vm["gpu_index"].as<int>();
    std::string dataset = vm["dataset"].as<std::string>();
    bool only_fuse = vm["only_fuse"].as<bool>();
    bool no_fuse = vm["no_fuse"].as<bool>();
    bool use_memory_cache = vm["memory_cache"].as<bool>();
    bool use_sa = vm["use_sa"].as<bool>();
    bool use_impetus = vm["use_impetus"].as<bool>();
    bool weak_filter = vm["weak_filter"].as<bool>();
    bool flush = vm["flush"].as<bool>();
    bool export_anchor = vm["export_anchor"].as<bool>();
    bool export_curve = vm["export_curve"].as<bool>();
    bool export_color = vm["export_color"].as<bool>();
    // it is not necessary to use memory cache when only_fuse is true
    if (only_fuse) {
        use_memory_cache = false;
    }
    // it is necessary to flush memory cache to disk when no_fuse is true
    if (no_fuse) {
        flush = true;
    }
    // show config
    std::cout << "========================== Config ==========================" << std::endl;
    std::cout << "dense_folder : " << dense_folder_str << std::endl;
    std::cout << "gpu_index    : " << gpu_index << std::endl;
    std::cout << "dataset      : " << dataset << std::endl;
    std::cout << "only_fuse    : " << only_fuse << std::endl;
    std::cout << "no_fuse      : " << no_fuse << std::endl;
    std::cout << "memory_cache : " << use_memory_cache << std::endl;
    std::cout << "use_sa       : " << use_sa << std::endl;
    std::cout << "use_impetus  : " << use_impetus << std::endl;
    std::cout << "weak_filter  : " << weak_filter << std::endl;
    std::cout << "flush        : " << flush << std::endl;
    std::cout << "export_anchor: " << export_anchor << std::endl;
    std::cout << "export_curve : " << export_curve << std::endl;
    std::cout << "export_color : " << export_color << std::endl;
    std::cout << "============================================================" << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // set environment for processing
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // use memory cache can speed up the program, but it will load all data into the memory
    // including all images, depths, normals, weaks, confidences, selected_views, cameras.
    if (use_memory_cache) {
        printf("Use memory cache!\n");
        auto memory_cache = MemoryCache::get_instance();
    }
    path dense_folder(dense_folder_str);
    path output_folder = dense_folder / path("APD");
    create_directory(output_folder);
    cudaSetDevice(gpu_index);
    // record the effective run-time options next to the results so every cloud can be traced to its settings
    {
        const std::string effective = registry.ToIni();
        std::cout << "==================== Run-time options ======================" << std::endl;
        std::cout << effective;
        std::cout << "============================================================" << std::endl;
        // A fusion-only run keeps the file of the PatchMatch run that produced the depth maps intact.
        const char *effective_name = only_fuse ? "apd_effective_config_fuse.ini" : "apd_effective_config.ini";
        std::ofstream effective_file((output_folder / path(effective_name)).string().c_str());
        effective_file << effective;
    }
    // the projection model must be known before the first camera file is read
    SetCameraModel(pipeline);
    // generate problems
    std::vector<Problem> problems;
    GenerateSampleList(dense_folder, problems, pipeline);
    if (!CheckImages(problems)) {
        std::cout << "Images may error, check it!\n";
        return EXIT_FAILURE;
    }
    std::cout << "There are " << problems.size() << " problems needed to be processed!" << std::endl;
    {
        const std::vector<std::string> changed = registry.ChangedOptions("weakfilter");
        if (std::find(changed.begin(), changed.end(), "weakfilter.confidence_as_uchar") != changed.end()) {
            std::cout << "WARNING: weakfilter.confidence_as_uchar is deprecated and has no effect (confidence maps "
                      << "are always read as stored)" << std::endl;
        }
    }
    // the Tanks and Temples fusion variants keep their own hard-coded thresholds
    if ((only_fuse || !no_fuse) && (dataset == "TaT_a" || dataset == "TaT_i")) {
        const char *groups[] = {"fusion", "weakfilter"};
        for (const char *group: groups) {
            if (!weak_filter && std::string(group) == "weakfilter")
                continue;                                     // the weak filter does not run at all
            const std::vector<std::string> changed = registry.ChangedOptions(group);
            for (const std::string &name: changed) {
                if (name == "weakfilter.confidence_as_uchar")
                    continue;                                 // already reported as deprecated
                std::cout << "WARNING: " << name << " is set but --dataset " << dataset
                          << " uses the Tanks and Temples fusion, which ignores it" << std::endl;
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // if only_fuse is true, then only do fusion
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (only_fuse) {
        if (dataset == "TaT_a") {
            RunFusion_TAT_A(dense_folder, problems, "APD.ply",  weak_filter, export_color);
        } else if (dataset == "TaT_i") {
            RunFusion_TAT_I(dense_folder, problems, "APD.ply",  weak_filter, export_color);
        } else {
            RunFusion(dense_folder, problems, "APD.ply",  weak_filter, export_color, config.fusion,
                      config.weak_filter);
        }
        printf("Fusion done!\n");
        return EXIT_SUCCESS;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // compute round num and set params
    ////////////////////////////////////////////////////////////////////////////////////////////////
    int round_num = ComputeRoundNum(problems, pipeline);
    std::cout << "Round nums: " << round_num << std::endl;
    if (pipeline.geom_iterations == 0 && (round_num > 1 || !no_fuse)) {
        // confidence.bin is written only by the geometric passes, and every later scale and the fusion read it
        std::cout << "ERROR: pipeline.geom_iterations = 0 writes no confidence.bin; it is only valid with one "
                  << "scale (e.g. pipeline.rounds = 1) and --no_fuse true (this run: " << round_num << " scale(s), "
                  << (no_fuse ? "no fusion" : "fusion") << ")" << std::endl;
        return EXIT_FAILURE;
    }
    // init common problem params
    for (auto &problem: problems) {
        problem.params = config.pm;  // start from the run-time options; the schedule below sets the rest
        problem.range_scale_min = pipeline.range_scale_min;
        problem.range_scale_max = pipeline.range_scale_max;
        problem.range_pad = pipeline.range_pad;
        if (pipeline.geom_factor >= 0.0f) {
            problem.params.geom_factor = pipeline.geom_factor;
        } else if (dataset == "TaT_a" || dataset == "TaT_i") {
            problem.params.geom_factor = 0.05f;
        } else {
            problem.params.geom_factor = 0.2f;
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // iteration for each round
    ////////////////////////////////////////////////////////////////////////////////////////////////
    int iteration_index = 0;
    const int geom_iteration = pipeline.geom_iterations;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < round_num; ++i) {
        std::cout << "========================== Round " << i << " ==========================" << std::endl;
        std::cout << "======== iteration " << iteration_index << "========" << std::endl;
        for (auto &problem: problems) {
            {
                auto &params = problem.params;
                if (i == 0) {
                    params.state = FIRST_INIT;
                    params.use_APD = false;
                } else {
                    params.state = REFINE_INIT;
                    params.use_APD = true;
                    params.ransac_threshold = pipeline.ransac_base - i * pipeline.ransac_step;
                    params.rotate_time = MIN(static_cast<int>(std::pow(2, i)), pipeline.rotate_time_max);
                }
                params.geom_consistency = false;
                params.max_iterations = pipeline.max_iterations;
                params.weak_peak_radius = pipeline.init_weak_peak_radius;
                params.use_sa = use_sa;
                params.use_impetus = use_impetus;
            }
            problem.show_medium_result = false;
            problem.iteration = iteration_index;
            problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale
            ProcessProblem(problem);
        }
        iteration_index++;
        for (int j = 0; j < geom_iteration; ++j) {
            std::cout << "======== iteration " << iteration_index << "========" << std::endl;
            bool is_last_iteration = (i == round_num - 1 && j == geom_iteration - 1);
            for (auto &problem: problems) {
                {
                    auto &params = problem.params;
                    params.state = REFINE_ITER;
                    if (i == 0) {
                        params.use_APD = false;
                    } else {
                        params.use_APD = true;
                        params.ransac_threshold = pipeline.ransac_base - i * pipeline.ransac_step;
                        params.rotate_time = MIN(static_cast<int>(std::pow(2, i)), pipeline.rotate_time_max);
                    }
                    params.geom_consistency = true;
                    params.max_iterations = pipeline.max_iterations;
                    params.weak_peak_radius = MAX(pipeline.weak_peak_start - pipeline.weak_peak_step * j,
                                                  pipeline.weak_peak_min);
                    params.use_sa = use_sa;
                    params.use_impetus = use_impetus;
                }
                if (is_last_iteration && export_anchor) {
                    problem.export_anchor = true;
                }
                if (is_last_iteration && export_curve) {
                    problem.export_reliable_curve = true;
                }
                problem.show_medium_result = (j == geom_iteration - 1);
                problem.iteration = iteration_index;
                problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale
                ProcessProblem(problem);
            }
            iteration_index++;
        }
        std::cout << "=============================================================" << std::endl;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms"
              << std::endl;

    int avg_used_time = 0;
    for (auto &problem: problems) {
        avg_used_time += problem.used_time;
    }
    avg_used_time /= problems.size();
    std::cout << "Average used time: " << avg_used_time << " ms" << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // flush memory cache to disk
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (use_memory_cache && flush) {
        printf("Write memory cache to disk!\n");
        auto memory_cache = MemoryCache::get_instance();
        for (auto &mat_info: memory_cache->mat_cache) {
            auto &mat_path = mat_info.first;
            auto &mat = mat_info.second;
            WriteBinMat(mat_path, mat, true);
        }
        printf("All done!\n");
        memory_cache->mat_cache.clear();
        memory_cache->img_cache.clear();
        memory_cache->cam_cache.clear();
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // fusion or not
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (no_fuse) {
        printf("Skip fusion, all done!\n");
        return EXIT_SUCCESS;
    }
    std::cout << "Run fusion\n";
    if (dataset == "TaT_a") {
        RunFusion_TAT_A(dense_folder, problems, "APD.ply", weak_filter, export_color);
    } else if (dataset == "TaT_i") {
        RunFusion_TAT_I(dense_folder, problems, "APD.ply", weak_filter, export_color);
    } else {
        RunFusion(dense_folder, problems, "APD.ply", weak_filter, export_color, config.fusion, config.weak_filter);
    }
    std::cout << "All done\n";
    return EXIT_SUCCESS;
}
