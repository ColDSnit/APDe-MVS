#include "APD.h"
#include <algorithm>
#include <cfloat>
#include <cmath>

static std::shared_ptr<MemoryCache> memory_cache = nullptr;
static std::mutex memory_cache_mutex;

// only call in the main.cpp
std::shared_ptr<MemoryCache> MemoryCache::get_instance() {
    if (memory_cache == nullptr) {
        std::unique_lock<std::mutex> lock(memory_cache_mutex);
        if (memory_cache == nullptr) {
            auto temp = std::shared_ptr<MemoryCache>(new MemoryCache());
            memory_cache = temp;
        }
    }
    return memory_cache;
}

bool ReadBinMat(const path &mat_path, cv::Mat &mat) {
    if (memory_cache != nullptr) {
        auto &mat_cache = memory_cache->mat_cache;
        if (mat_cache.find(mat_path.string()) != mat_cache.end()) {
            mat = mat_cache[mat_path.string()].clone();
            return true;
        }
    }
    ifstream in(mat_path, std::ios_base::binary);
    if (!in.is_open()) {                                     // missing or unreadable (bad() is not set by a failed open)
        std::cout << "Error opening file: " << mat_path << std::endl;
        return false;
    }

    int version = 0, rows = 0, cols = 0, type = 0;           // defined values when the header is short
    in.read((char *) (&version), sizeof(int));
    in.read((char *) (&rows), sizeof(int));
    in.read((char *) (&cols), sizeof(int));
    in.read((char *) (&type), sizeof(int));

    if (!in) {
        in.close();
        std::cout << "Error: " << mat_path << " is shorter than its 16-byte header" << std::endl;
        return false;
    }
    if (version != 1) {
        in.close();
        std::cout << "Version error: " << mat_path << std::endl;
        return false;
    }
    if (rows < 0 || cols < 0 || (type & ~CV_MAT_TYPE_MASK) != 0) {
        in.close();
        std::cout << "Error: " << mat_path << " has an invalid header (" << rows << " x " << cols << ", type " << type
                  << ")" << std::endl;
        return false;
    }
    // the data the header announces must all be in the file; a shorter file is rejected like a missing one, before
    // anything is allocated (a corrupt header cannot trigger a huge allocation)
    const std::streampos data_start = in.tellg();
    in.seekg(0, std::ios_base::end);
    const std::streampos data_end = in.tellg();
    in.seekg(data_start);
    if (!in || data_start < 0 || data_end < data_start) {
        in.close();
        std::cout << "Error: cannot determine the size of " << mat_path << std::endl;
        return false;
    }
    const long long available_bytes = (long long) (data_end - data_start);
    const long long elem_bytes = CV_ELEM_SIZE(type);
    // rows * cols * elem_bytes <= available_bytes, tested by division so that no product can overflow
    const bool fits = rows == 0 || cols == 0 ||
                      (cols <= available_bytes / elem_bytes && rows <= available_bytes / (cols * elem_bytes));
    if (!fits) {
        in.close();
        std::cout << "Error: " << mat_path << " is shorter than its header says (" << rows << " x " << cols
                  << ", type " << type << ", " << available_bytes << " data bytes); not read" << std::endl;
        return false;
    }
    const long long expected_bytes = (long long) rows * cols * elem_bytes;

    // a freshly constructed matrix, so a buffer the caller's mat shares with another Mat is never written
    cv::Mat data(rows, cols, type);
    in.read((char *) data.data, (std::streamsize) expected_bytes);
    const std::streamsize read_bytes = in.gcount();
    if (read_bytes != (std::streamsize) expected_bytes) {
        in.close();
        std::cout << "Error: reading " << mat_path << " stopped after " << read_bytes << " of " << expected_bytes
                  << " data bytes; not read" << std::endl;
        return false;
    }
    in.close();
    mat = data;
    if (memory_cache != nullptr) {
        auto &mat_cache = memory_cache->mat_cache;
        if (mat_cache.find(mat_path.string()) == mat_cache.end()) {
            mat_cache[mat_path.string()] = mat.clone();
        } else {
            printf("Error: %s already exists in mat_cache\n", mat_path.string().c_str());
        }
    }
    return true;
}

bool WriteBinMat(const path &mat_path, const cv::Mat &mat, bool flush) {
    if (memory_cache != nullptr) {
        auto &mat_cache = memory_cache->mat_cache;
        mat_cache[mat_path.string()] = mat.clone();
    }

    if (flush || memory_cache == nullptr) {
        ofstream out(mat_path, std::ios_base::binary);
        if (out.bad()) {
            std::cout << "Error opening file: " << mat_path << std::endl;
            return false;
        }
        int version = 1;
        int rows = mat.rows;
        int cols = mat.cols;
        int type = mat.type();

        out.write((char *) &version, sizeof(int));
        out.write((char *) &rows, sizeof(int));
        out.write((char *) &cols, sizeof(int));
        out.write((char *) &type, sizeof(int));
        out.write((char *) mat.data, sizeof(char) * mat.step * mat.rows);
        out.close();
    }
    return true;
}

// Projection model applied to every camera that is read (pinhole unless main() says otherwise).
static PipelineParams camera_model_params;

void SetCameraModel(const PipelineParams &pipeline) {
    camera_model_params = pipeline;
}

bool ReadCamera(const path &cam_path, Camera &cam) {
    if (memory_cache != nullptr) {
        auto &cam_cache = memory_cache->cam_cache;
        if (cam_cache.find(cam_path.string()) != cam_cache.end()) {
            cam = cam_cache[cam_path.string()];
            return true;
        }
    }

    ifstream in(cam_path);
    if (!in.is_open()) {                                     // missing or unreadable (bad() is not set by a failed open)
        return false;
    }

    std::string line;
    in >> line;

    for (int i = 0; i < 3; ++i) {
        in >> cam.R[3 * i + 0] >> cam.R[3 * i + 1] >> cam.R[3 * i + 2] >> cam.t[i];
    }

    float tmp[4];
    in >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
    in >> line;

    for (int i = 0; i < 3; ++i) {
        in >> cam.K[3 * i + 0] >> cam.K[3 * i + 1] >> cam.K[3 * i + 2];
    }
    if (!in) {
        return false;                                        // file ends before the intrinsics are complete
    }
    // compute camera center in world coord
    const auto &R = cam.R;
    const auto &t = cam.t;
    for (int j = 0; j < 3; ++j) {
        cam.c[j] = -float(
                double(R[0 + j]) * double(t[0]) + double(R[3 + j]) * double(t[1]) + double(R[6 + j]) * double(t[2]));
    }
    in >> cam.depth_min >> cam.interval;
    if (!(in >> cam.depth_num >> cam.depth_max)) {
        cam.depth_num = 192;
        cam.depth_max = cam.interval * cam.depth_num + cam.depth_min;
    }
    in.close();
    cam.model = camera_model_params.camera_model;
    if (cam.model == CAMERA_ORTHOGRAPHIC) {
        // The pinhole-equivalent focal length is exact at one depth only; by default that is taken to be the
        // middle of this camera's depth search range, i.e. where the object is.
        cam.ref_depth = camera_model_params.ortho_ref_depth > 0.0f
                        ? camera_model_params.ortho_ref_depth
                        : 0.5f * (cam.depth_min * camera_model_params.range_scale_min +
                                  cam.depth_max * camera_model_params.range_scale_max);
    }
    if (memory_cache != nullptr) {
        auto &cam_cache = memory_cache->cam_cache;
        if (cam_cache.find(cam_path.string()) == cam_cache.end()) {
            cam_cache[cam_path.string()] = cam;
        } else {
            printf("Error: %s already exists in cam_cache\n", cam_path.string().c_str());
        }
    }
    return true;
}

bool ReadImage(const path &img_path, cv::Mat &img) {
    if (memory_cache != nullptr) {
        auto &img_cache = memory_cache->img_cache;
        if (img_cache.find(img_path.string()) != img_cache.end()) {
            img = img_cache[img_path.string()].clone();
            return true;
        }
    }
    cv::Mat_<uint8_t> image_uint = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);
    if (image_uint.empty()) {
        std::cout << "Error opening file: " << img_path << std::endl;
        return false;
    }
    image_uint.convertTo(img, CV_32FC1);
    if (memory_cache != nullptr) {
        auto &img_cache = memory_cache->img_cache;
        if (img_cache.find(img_path.string()) == img_cache.end()) {
            img_cache[img_path.string()] = img.clone();
        } else {
            printf("Error: %s already exists in img_cache\n", img_path.string().c_str());
        }
    }
    return true;
}

bool ShowDepthMap(const path &depth_path, const cv::Mat &depth, float depth_min, float depth_max) {
    // compute depth mean
    float depth_mean = 0;
    int depth_mean_count = 0;
    for (int i = 0; i < depth.cols; i++) {
        float depth_mean_row_sum = 0;
        int depth_mean_row_count = 0;
        for (int j = 0; j < depth.rows; j++) {
            if (depth.at<float>(j, i) < depth_min || depth.at<float>(j, i) > depth_max ||
                isnan(depth.at<float>(j, i))) {
                continue;
            }
            depth_mean_row_sum += depth.at<float>(j, i);
            depth_mean_row_count++;
        }
        if (depth_mean_row_count > 0) {
            depth_mean += depth_mean_row_sum / depth_mean_row_count;
            depth_mean_count++;
        }
    }
    if (depth_mean_count > 0) {
        depth_mean /= depth_mean_count;
    }
    // compute depth std
    float depth_std = 0;
    int depth_std_count = 0;
    for (int i = 0; i < depth.cols; i++) {
        float depth_std_row_sum = 0;
        int depth_std_row_count = 0;
        for (int j = 0; j < depth.rows; j++) {
            if (depth.at<float>(j, i) < depth_min || depth.at<float>(j, i) > depth_max ||
                isnan(depth.at<float>(j, i))) {
                continue;
            }
            depth_std_row_sum += pow(depth.at<float>(j, i) - depth_mean, 2);
            depth_std_row_count++;
        }
        if (depth_std_row_count > 0) {
            depth_std += depth_std_row_sum / depth_std_row_count;
            depth_std_count++;
        }
    }
    if (depth_std_count > 0) {
        depth_std = sqrt(depth_std / depth_std_count);
    }
    // compute depth min and max
    float depth_val_min = depth_mean - depth_std * 2;
    float depth_val_max = depth_mean + depth_std * 2;
    float depth_delta = depth_val_max - depth_val_min;

    // convert depth to gray
    cv::Mat depth_gray(depth.size(), CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < depth.cols; i++) {
        for (int j = 0; j < depth.rows; j++) {
            float pixel_val = (depth.at<float>(j, i) - depth_val_min) / depth_delta;
            if (pixel_val > 1) {
                pixel_val = 1;
            }
            if (pixel_val < 0) {
                pixel_val = 0;
            }
            pixel_val = pixel_val * 255;
            depth_gray.at<uchar>(j, i) = static_cast<uchar>(pixel_val);
        }
    }
    cv::Mat result_img(depth.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    // to jet color
    cv::applyColorMap(depth_gray, result_img, cv::COLORMAP_JET);
    cv::imwrite(depth_path.string(), result_img);
    return true;
}


bool ShowConfidenceMap(const path &confidence_path, const cv::Mat &confidence) {
    uchar max_c = 0;
    uchar min_c = 255;
    for (int r = 0; r < confidence.rows; r++) {
        for (int c = 0; c < confidence.cols; c++) {
            uchar val = confidence.at<uchar>(r, c);
            if (val > max_c) {
                max_c = val;
            }
            if (val < min_c) {
                min_c = val;
            }
        }
    }
    uchar delta_c = max_c - min_c;
    if (delta_c == 0) {
        delta_c = 1;
    }
    cv::Mat result_img(confidence.size(), CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < confidence.rows; r++) {
        for (int c = 0; c < confidence.cols; c++) {
            uchar val = confidence.at<uchar>(r, c);
            uchar pixel_val = (val - min_c) * 255 / delta_c;
            result_img.at<uchar>(r, c) = pixel_val;
        }
    }
    cv::imwrite(confidence_path.string(), result_img);
    return true;
}

bool ShowNormalMap(const path &normal_path, const cv::Mat &normal) {
    if (normal.empty()) {
        return false;
    }
    cv::Mat normalized_normal = normal.clone();
    for (int i = 0; i < normalized_normal.rows; i++) {
        for (int j = 0; j < normalized_normal.cols; j++) {
            cv::Vec3f normal_val = normalized_normal.at<cv::Vec3f>(i, j);
            float norm = sqrt(pow(normal_val[0], 2) + pow(normal_val[1], 2) + pow(normal_val[2], 2));
            if (norm == 0) {
                normalized_normal.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0);
            } else {
                normalized_normal.at<cv::Vec3f>(i, j) = normal_val / norm;
            }
        }
    }

    cv::Mat img(normalized_normal.size(), CV_8UC3, cv::Scalar(0.f, 0.f, 0.f));
    normalized_normal.convertTo(img, img.type(), 255.f / 2.f, 255.f / 2.f);
    cv::imwrite(normal_path.string(), img);
    return true;
}

bool ShowWeakImage(const path &weak_path, const cv::Mat &weak) {
    // show image
    if (weak.empty()) {
        return false;
    }
    const int width = weak.cols;
    const int height = weak.rows;
    cv::Mat weak_info_image(height, width, CV_8UC3);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            switch (weak.at<uchar>(r, c)) {
                case WEAK:
                    weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
                    break;
                case STRONG:
                    weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 0);
                    break;
                case UNKNOWN:
                    weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 255);
                    break;
            }
        }
    }
    // save
    cv::imwrite(weak_path.string(), weak_info_image);
    return true;
}

bool ExportPointCloud(const path &point_cloud_path, std::vector<PointList> &pointcloud, bool export_color) {
    ofstream out(point_cloud_path, std::ios::binary);
    if (out.bad()) {
        return false;
    }

    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "element vertex " << int(pointcloud.size()) << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    if (export_color) {
        out << "property uchar blue\n";
        out << "property uchar green\n";
        out << "property uchar red\n";
    }
    out << "end_header\n";

    for (size_t idx = 0; idx < pointcloud.size(); idx++) {
        float px = pointcloud[idx].coord.x;
        float py = pointcloud[idx].coord.y;
        float pz = pointcloud[idx].coord.z;

        out.write((char *) &px, sizeof(float));
        out.write((char *) &py, sizeof(float));
        out.write((char *) &pz, sizeof(float));

        if (export_color) {
            cv::Vec3b pixel;
            pixel[0] = static_cast<uchar>(pointcloud[idx].color.x);
            pixel[1] = static_cast<uchar>(pointcloud[idx].color.y);
            pixel[2] = static_cast<uchar>(pointcloud[idx].color.z);
            out.write((char *) &pixel[0], sizeof(uchar));
            out.write((char *) &pixel[1], sizeof(uchar));
            out.write((char *) &pixel[2], sizeof(uchar));
        }
    }
    out.close();
    return true;
}

void StringAppendV(std::string *dst, const char *format, va_list ap) {
    // First try with a small fixed size buffer.
    static const int kFixedBufferSize = 1024;
    char fixed_buffer[kFixedBufferSize];

    // It is possible for methods that use a va_list to invalidate
    // the data in it upon use.  The fix is to make a copy
    // of the structure before using it and use that copy instead.
    va_list backup_ap;
    va_copy(backup_ap, ap);
    int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
    va_end(backup_ap);

    if (result < kFixedBufferSize) {
        if (result >= 0) {
            // Normal case - everything fits.
            dst->append(fixed_buffer, result);
            return;
        }

#ifdef _MSC_VER
        // Error or MSVC running out of space.  MSVC 8.0 and higher
        // can be asked about space needed with the special idiom below:
        va_copy(backup_ap, ap);
        result = vsnprintf(nullptr, 0, format, backup_ap);
        va_end(backup_ap);
#endif

        if (result < 0) {
            // Just an error.
            return;
        }
    }

    // Increase the buffer size to the size requested by vsnprintf,
    // plus one for the closing \0.
    const int variable_buffer_size = result + 1;
    std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

    // Restore the va_list before we use it again.
    va_copy(backup_ap, ap);
    result =
            vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
    va_end(backup_ap);

    if (result >= 0 && result < variable_buffer_size) {
        dst->append(variable_buffer.get(), result);
    }
}

std::string StringPrintf(const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    std::string result;
    StringAppendV(&result, format, ap);
    va_end(ap);
    return result;
}

void CudaSafeCall(const cudaError_t error, const std::string &file,
                  const int line) {
    if (error != cudaSuccess) {
        std::cout << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
                                  file.c_str(), line)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CudaCheckError(const char *file, const int line) {
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
                                  line, cudaGetErrorString(error))
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    error = cudaDeviceSynchronize();
    if (cudaSuccess != error) {
        std::cout << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
                                  file, line, cudaGetErrorString(error))
                  << std::endl;
        std::cout
                << "This error is likely caused by the graphics card timeout "
                   "detection mechanism of your operating system. Please refer to "
                   "the FAQ in the documentation on how to solve this problem."
                << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::string ToFormatIndex(int index) {
    std::stringstream ss;
    ss << std::setw(8) << std::setfill('0') << index;
    return ss.str();
}

APD::APD(const Problem &problem) {
    params_host = problem.params;
    this->problem = problem;
}

APD::~APD() {
    // free images
    {
        for (int i = 0; i < num_images; ++i) {
            cudaDestroyTextureObject(texture_objects_host.images[i]);
            cudaFreeArray(cuArray[i]);
        }
        cudaFree(texture_objects_cuda);
    }
    // may free depths
    if (params_host.geom_consistency || params_host.use_APD) {
        for (int i = 0; i < num_images; ++i) {
            cudaDestroyTextureObject(texture_depths_host.images[i]);
            cudaFreeArray(cuDepthArray[i]);
        }
        cudaFree(texture_depths_cuda);
    }
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(params_cuda);
    cudaFree(helper_cuda);
    cudaFree(weak_info_cuda);
    cudaFree(view_weight_cuda);
    cudaFree(confidence_cuda);
    cudaFree(sa_mask_cuda);
    if (prior_bounds_cuda != nullptr) {
        cudaFree(prior_bounds_cuda);
    }

    if (params_host.use_APD) {
        cudaFree(fit_plane_hypotheses_cuda);
        cudaFree(weak_reliable_cuda);
        cudaFree(weak_nearest_strong_cuda);
        cudaFree(anchors_cuda);
        cudaFree(anchors_map_cuda);
    }
}

// PatchMatch cannot run without one of its inputs: a missing or corrupt image, cam file or intermediate map stops
// APD with a message naming the file (upstream went on and crashed in cv::resize or ran on an unset camera).
static void StopOnInput(const std::string &problem_text, const path &file) {
    std::cout << "ERROR: " << problem_text << ": " << file << " - APD stops" << std::endl;
    exit(EXIT_FAILURE);
}

// Reads a BinMat that PatchMatch needs, of the given OpenCV type; stops APD when it is missing, empty or of
// another type.
static void ReadRequiredBinMat(const path &mat_path, int type, cv::Mat &mat) {
    if (!ReadBinMat(mat_path, mat) || mat.empty()) {
        StopOnInput("missing, truncated or empty map", mat_path);
    }
    if (mat.type() != type) {
        StopOnInput("map of OpenCV type " + std::to_string(mat.type()) + " (expected " + std::to_string(type) + ")",
                    mat_path);
    }
}

void APD::InuputInitialization() {
    images.clear();
    cameras.clear();
    // get folder
    path image_folder = problem.dense_folder / path("images");
    path cam_folder = problem.dense_folder / path("cams");
    path sa_mask_folder = problem.dense_folder / path("sa_masks");
    // =================================================
    // read ref image and src images
    // ref
    {
        path ref_image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + problem.img_ext);
        cv::Mat image_float;
        if (!ReadImage(ref_image_path, image_float)) {
            StopOnInput("cannot read the reference image", ref_image_path);
        }
        images.push_back(image_float);
        width = image_float.cols;
        height = image_float.rows;
    }
    // src
    for (const auto &src_idx: problem.src_image_ids) {
        path src_image_path = image_folder / path(ToFormatIndex(src_idx) + problem.img_ext);
        cv::Mat image_float;
        if (!ReadImage(src_image_path, image_float)) {
            StopOnInput("cannot read a source image", src_image_path);
        }
        if (image_float.cols != width || image_float.rows != height) {
            StopOnInput("source image size differs from the reference image (" + std::to_string(width) + " x " +
                        std::to_string(height) + ")", src_image_path);
        }
        images.push_back(image_float);
    }
    if (images.size() > MAX_IMAGES) {
        std::cout << "Can't process so much images: " << images.size() << std::endl;
        exit(EXIT_FAILURE);
    }
    // =================================================
    // read ref camera and src camera
    // ref
    {
        path ref_cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
        Camera cam;
        if (!ReadCamera(ref_cam_path, cam)) {
            StopOnInput("cannot read the cam file", ref_cam_path);
        }
        cam.width = width;
        cam.height = height;
        cameras.push_back(cam);
    }
    // src
    for (const auto &src_idx: problem.src_image_ids) {
        path src_cam_path = cam_folder / path(ToFormatIndex(src_idx) + "_cam.txt");
        Camera cam;
        if (!ReadCamera(src_cam_path, cam)) {
            StopOnInput("cannot read the cam file", src_cam_path);
        }
        cam.width = width;
        cam.height = height;
        cameras.push_back(cam);
    }
    // =================================================
    // set some params
    params_host.depth_min = cameras[0].depth_min * problem.range_scale_min;
    params_host.depth_max = cameras[0].depth_max * problem.range_scale_max;
    if (problem.range_pad != 0.0f) {
        // absolute widening; useful when depths are metres away and the object is millimetres deep
        params_host.depth_min -= problem.range_pad;
        params_host.depth_max += problem.range_pad;
    }
    params_host.num_images = (int) images.size();
    num_images = (int) images.size();
    // =================================================
    std::cout << "Read images and camera done\n";
    std::cout << "Depth range: " << params_host.depth_min << " " << params_host.depth_max << std::endl;
    std::cout << "Num images: " << params_host.num_images << std::endl;
    // =================================================
    const int full_width = width;                                  // reference image size before scaling (prior size)
    const int full_height = height;
    // scale images
    if (problem.scale_size != 1) {
        for (int i = 0; i < num_images; ++i) {
            const float factor = 1.0f / (float) (problem.scale_size);
            const int new_cols = std::round(images[i].cols * factor);
            const int new_rows = std::round(images[i].rows * factor);

            const float scale_x = new_cols / static_cast<float>(images[i].cols);
            const float scale_y = new_rows / static_cast<float>(images[i].rows);

            cv::Mat_<float> scaled_image_float;
            cv::resize(images[i], scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
            images[i] = scaled_image_float.clone();

            width = scaled_image_float.cols;
            height = scaled_image_float.rows;

            cameras[i].K[0] *= scale_x;
            cameras[i].K[2] *= scale_x;
            cameras[i].K[4] *= scale_y;
            cameras[i].K[5] *= scale_y;
            cameras[i].width = width;
            cameras[i].height = height;
        }
        std::cout << "Scale images and cameras done\n";
    }
    std::cout << "Image size: " << width << " * " << height << std::endl;
    // =================================================
    // per-pixel depth prior (off by default)
    if (params_host.prior_enable) {
        LoadDepthPrior(full_width, full_height);
    }
    // =================================================
    // read depth form geom consistency
    if (params_host.geom_consistency || params_host.use_APD) {
        depths.clear();
        path ref_depth_path = problem.result_folder / path("depths.bin");
        cv::Mat ref_depth;
        ReadRequiredBinMat(ref_depth_path, CV_32FC1, ref_depth);
        depths.push_back(ref_depth);
        for (const auto &src_idx: problem.src_image_ids) {
            path src_depth_path =
                    problem.dense_folder / path("APD") / path(ToFormatIndex(src_idx)) / path("depths.bin");
            cv::Mat src_depth;
            ReadRequiredBinMat(src_depth_path, CV_32FC1, src_depth);
            depths.push_back(src_depth);
        }
        for (auto &depth: depths) {
            if (depth.cols != width || depth.rows != height) {
                cv::resize(depth, depth, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
            }
        }
    }
    // =================================================
    // read weak info
    sa_mask_host = cv::Mat::zeros(height, width, CV_8UC1);
    if (params_host.use_APD) {
        path weak_info_path = problem.result_folder / path("weak.bin");
        path confidence_path = problem.result_folder / path("confidence.bin");
        ReadRequiredBinMat(weak_info_path, CV_8UC1, weak_info_host);
        ReadRequiredBinMat(confidence_path, CV_8UC1, confidence_host);    // uploaded to the GPU as uchar
        if (weak_info_host.cols != width || weak_info_host.rows != height) {
            std::cout << "resize weak info to target size\n";
            cv::resize(weak_info_host, weak_info_host, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
        }
        if (confidence_host.cols != width || confidence_host.rows != height) {
            std::cout << "resize confidence to target size\n";
            cv::resize(confidence_host, confidence_host, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
        }
        anchors_map_host = cv::Mat::zeros(weak_info_host.size(), CV_32SC1);
        weak_count = 0;
        for (int r = 0; r < weak_info_host.rows; ++r) {
            for (int c = 0; c < weak_info_host.cols; ++c) {
                int val = weak_info_host.at<uchar>(r, c);
                // point is weak
                if (val == WEAK) {
                    anchors_map_host.at<int>(r, c) = weak_count;
                    weak_count++;
                } else {
                    anchors_map_host.at<int>(r, c) = -1;
                }
            }
        }
        if (params_host.use_sa) {
            if (exists(sa_mask_folder)) {
                path sa_mask_path = sa_mask_folder / path(ToFormatIndex(problem.ref_image_id) + ".bin");
                cv::Mat sa;
                if (!ReadBinMat(sa_mask_path, sa) || sa.empty() || sa.type() != CV_8UC1) {
                    // optional input (upstream also ran without it): this view runs without a sa mask
                    std::cout << "WARNING: no usable sa mask " << sa_mask_path
                              << " (missing, unreadable, empty or not CV_8UC1); none used for this view" << std::endl;
                } else {
                    sa_mask_host = sa;
                }
                if (sa_mask_host.cols != width || sa_mask_host.rows != height) {
                    std::cout << "resize sa mask to target size\n";
                    cv::resize(sa_mask_host, sa_mask_host, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
                }
            } else {
                std::cout << "Can't find sa mask folder: " << sa_mask_folder << std::endl;
            }
        }
        std::cout << "Weak count: " << weak_count << " / " << weak_info_host.cols * weak_info_host.rows << " = "
                  << (float) weak_count / (float) (weak_info_host.cols * weak_info_host.rows) * 100 << "%" << std::endl;
    } else {
        weak_info_host = cv::Mat(height, width, CV_8UC1, cv::Scalar(STRONG));
        confidence_host = cv::Mat::ones(height, width, CV_8UC1);
        weak_count = 0;
    }
    // =================================================
    plane_hypotheses_host.reset(new float4[width * height]());
    if (params_host.state != FIRST_INIT) {
        // input plane hypotheses from existed result
        path depth_path = problem.result_folder / path("depths.bin");
        path normal_path = problem.result_folder / path("normals.bin");
        cv::Mat depth, normal;
        ReadRequiredBinMat(depth_path, CV_32FC1, depth);
        ReadRequiredBinMat(normal_path, CV_32FC3, normal);
        if (depth.cols != width || depth.rows != height || normal.cols != width || normal.rows != height) {
            std::cout << "resize depth and normal to target size\n";
            cv::resize(depth, depth, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
            cv::resize(normal, normal, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
        }
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                plane_hypotheses_host[center].w = depth.at<float>(row, col);
                plane_hypotheses_host[center].x = normal.at<cv::Vec3f>(row, col)[0];
                plane_hypotheses_host[center].y = normal.at<cv::Vec3f>(row, col)[1];
                plane_hypotheses_host[center].z = normal.at<cv::Vec3f>(row, col)[2];
            }
        }
    }
    // =================================================
}

void APD::CudaSpaceInitialization() {
    // =================================================
    // move images to gpu
    for (int i = 0; i < num_images; ++i) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray[i], &channelDesc, width, height);
        cudaMemcpy2DToArray(cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], width * sizeof(float), height,
                            cudaMemcpyHostToDevice);
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
    }
    cudaMalloc((void **) &texture_objects_cuda, sizeof(cudaTextureObjects));
    cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);
    // may move depths to gpu
    if (params_host.geom_consistency || params_host.use_APD) {
        for (int i = 0; i < num_images; ++i) {
            int height = depths[i].rows;
            int width = depths[i].cols;
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuDepthArray[i], &channelDesc, width, height);
            cudaMemcpy2DToArray(cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], width * sizeof(float),
                                height, cudaMemcpyHostToDevice);
            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuDepthArray[i];
            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;
            cudaCreateTextureObject(&(texture_depths_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void **) &texture_depths_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);
    }
    // =================================================
    // move camera to gpu
    cudaMalloc((void **) &cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);
    // malloc memory for important data structure
    const int length = width * height;
    // define cost
    cudaMalloc((void **) &costs_cuda, sizeof(float) * length);
    // malloc memory for rand states
    cudaMalloc((void **) &rand_states_cuda, sizeof(curandState) * length);
    // malloc for selected_views
    cudaMalloc((void **) &selected_views_cuda, sizeof(unsigned int) * length);
    // cudaMemcpy(selected_views_cuda, selected_views_host.ptr<unsigned>(0),
    //            sizeof(unsigned) * length, cudaMemcpyHostToDevice);
    // view weight
    cudaMalloc((void **) &view_weight_cuda, sizeof(uchar) * length * MAX_IMAGES);
    // move plane hypotheses to gpu
    cudaMalloc((void **) &plane_hypotheses_cuda, sizeof(float4) * length);
    cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host.get(),
               sizeof(float4) * length, cudaMemcpyHostToDevice);
    // malloc memory for weak info
    cudaMalloc((void **) (&weak_info_cuda), length * sizeof(uchar));
    cudaMemcpy(weak_info_cuda, weak_info_host.ptr<uchar>(0),
               length * sizeof(uchar), cudaMemcpyHostToDevice);
    // malloc memory for confidence
    cudaMalloc((void **) (&confidence_cuda), length * sizeof(uchar));
    cudaMemcpy(confidence_cuda, confidence_host.ptr<uchar>(0),
               length * sizeof(uchar), cudaMemcpyHostToDevice);
    // malloc memory for sa mask
    cudaMalloc((void **) (&sa_mask_cuda), length * sizeof(uchar));
    cudaMemcpy(sa_mask_cuda, sa_mask_host.ptr<uchar>(0),
               length * sizeof(uchar), cudaMemcpyHostToDevice);
    // per-pixel depth bounds (only when a prior was loaded)
    if (!prior_bounds_host.empty()) {
        cudaMalloc((void **) (&prior_bounds_cuda), length * sizeof(float2));
        cudaMemcpy(prior_bounds_cuda, prior_bounds_host.ptr<float>(0), length * sizeof(float2), cudaMemcpyHostToDevice);
    }

    if (params_host.use_APD) {
        // malloc memory for fit plane
        cudaMalloc((void **) &fit_plane_hypotheses_cuda, sizeof(float4) * length);
        cudaMemset(fit_plane_hypotheses_cuda, 0, sizeof(float4) * length);
        // malloc memory for weak reliable info
        cudaMalloc((void **) (&weak_reliable_cuda), length * sizeof(uchar));
        // malloc memory for nearest strong points
        cudaMalloc((void **) (&weak_nearest_strong_cuda), length * sizeof(short2));
        // move anchor map to gpu
        cudaMalloc((void **) (&anchors_map_cuda), length * sizeof(int));
        cudaMemcpy(anchors_map_cuda, anchors_map_host.ptr<int>(0),
                   length * sizeof(int), cudaMemcpyHostToDevice);
        // malloc memory for deformable ncc
        cudaMalloc((void **) (&anchors_cuda), weak_count * ANCHOR_NUM * sizeof(short2));
    }
    // malloc memory for anchor cluster color
    // move param to gpu
    cudaMalloc((void **) (&params_cuda), sizeof(PatchMatchParams));
    cudaMemcpy(params_cuda, &params_host, sizeof(PatchMatchParams), cudaMemcpyHostToDevice);
    // =================================================
}

void APD::SetDataPassHelperInCuda() {
    helper_host.width = this->width;
    helper_host.height = this->height;
    helper_host.ref_index = this->problem.ref_image_id;
    helper_host.texture_depths_cuda = this->texture_depths_cuda;
    helper_host.texture_objects_cuda = this->texture_objects_cuda;
    helper_host.cameras_cuda = this->cameras_cuda;
    helper_host.costs_cuda = this->costs_cuda;
    helper_host.anchors_cuda = this->anchors_cuda;
    helper_host.anchors_map_cuda = this->anchors_map_cuda;
    helper_host.plane_hypotheses_cuda = this->plane_hypotheses_cuda;
    helper_host.rand_states_cuda = this->rand_states_cuda;
    helper_host.selected_views_cuda = this->selected_views_cuda;
    helper_host.weak_info_cuda = this->weak_info_cuda;
    helper_host.confidence_cuda = this->confidence_cuda;
    helper_host.sa_mask_cuda = this->sa_mask_cuda;
    helper_host.params = this->params_cuda;
    helper_host.debug_point = make_int2(DEBUG_POINT_X, DEBUG_POINT_Y);
    helper_host.fit_plane_hypotheses_cuda = this->fit_plane_hypotheses_cuda;
    helper_host.weak_reliable_cuda = this->weak_reliable_cuda;
    helper_host.view_weight_cuda = this->view_weight_cuda;
    helper_host.weak_nearest_strong = this->weak_nearest_strong_cuda;
    helper_host.prior_bounds_cuda = this->prior_bounds_cuda;
    cudaMalloc((void **) (&helper_cuda), sizeof(DataPassHelper));
    cudaMemcpy(helper_cuda, &helper_host, sizeof(DataPassHelper), cudaMemcpyHostToDevice);
}

float4 APD::GetPlaneHypothesis(int r, int c) {
    return plane_hypotheses_host[c + r * width];
}

cv::Mat APD::GetPixelStates() {
    return weak_info_host;
}

cv::Mat APD::GetConfidence() {
    return confidence_host;
}

int APD::GetWidth() {
    return width;
}

int APD::GetHeight() {
    return height;
}

float APD::GetDepthMin() {
    return params_host.depth_min;
}

float APD::GetDepthMax() {
    return params_host.depth_max;
}

bool APD::DepthInBounds(int r, int c, float depth) {
    float lo = params_host.depth_min;                          // default: the view's search range
    float hi = params_host.depth_max;
    if (!prior_bounds_host.empty()) {
        const cv::Vec2f b = prior_bounds_host.at<cv::Vec2f>(r, c);
        if (b[1] > b[0]) {                                     // valid prior band replaces the view range
            lo = b[0];
            hi = b[1];
        }
    }
    return !(depth < lo || depth > hi);                        // upstream's test: a NaN depth is not rejected here
}

// Reads a depth-prior BinMat without the memory cache (a cached input would be written back to disk by --flush):
// checks the header, the type (CV_32FC1 / CV_32FC2), the size (must be expected_cols x expected_rows) and the file
// length before reading the data, and the read itself.
static bool ReadDepthPriorFile(const path &prior_path, int expected_cols, int expected_rows, cv::Mat &prior,
                               std::string &problem_text) {
    boost::filesystem::ifstream in(prior_path, std::ios_base::binary);      // wide path on Windows, as ReadBinMat
    if (!in.is_open()) {
        problem_text = "cannot be opened";
        return false;
    }
    int header[4] = {0, 0, 0, 0};                              // version, rows, cols, OpenCV type
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    if (in.gcount() != (std::streamsize) sizeof(header) || header[0] != 1) {
        problem_text = "has no valid BinMat header";
        return false;
    }
    const int rows = header[1], cols = header[2], type = header[3];
    if (type != CV_32FC1 && type != CV_32FC2) {
        problem_text = "has OpenCV type " + std::to_string(type) + " (expected CV_32FC1 depth or CV_32FC2 lo/hi)";
        return false;
    }
    if (rows != expected_rows || cols != expected_cols) {      // also bounds the size arithmetic below
        problem_text = "is " + std::to_string(cols) + "x" + std::to_string(rows) + ", the image is " +
                       std::to_string(expected_cols) + "x" + std::to_string(expected_rows);
        return false;
    }
    const uintmax_t data_bytes = (uintmax_t) rows * (uintmax_t) cols * (uintmax_t) CV_ELEM_SIZE(type);
    boost::system::error_code size_error;
    const uintmax_t bytes = file_size(prior_path, size_error);
    if (size_error || bytes != sizeof(header) + data_bytes) {
        problem_text = size_error ? "has no readable size" : "has " + std::to_string(bytes) +
                       " bytes, the header needs " + std::to_string(sizeof(header) + data_bytes);
        return false;
    }
    prior = cv::Mat(rows, cols, type);
    in.read(reinterpret_cast<char *>(prior.data), (std::streamsize) data_bytes);
    if (in.gcount() != (std::streamsize) data_bytes) {
        problem_text = "could not be read completely";
        prior.release();
        return false;
    }
    return true;
}

void APD::LoadDepthPrior(int full_width, int full_height) {
    // The prior is stored once at the reference image's own size (before APD's scaling): CV_32FC1 = prior depth
    // (band from the prior.band_near / prior.band_far options) or CV_32FC2 = (lo, hi), camera-frame z in world units.
    // 0 / non-finite / hi <= lo = no prior at that pixel.
    path prior_path = problem.dense_folder / path("depth_prior") / path(ToFormatIndex(problem.ref_image_id) + ".bin");
    if (!exists(prior_path)) {
        std::cout << "WARNING Depth prior missing (view range used everywhere): " << prior_path << std::endl;
        return;
    }
    cv::Mat prior;
    std::string problem_text;
    if (!ReadDepthPriorFile(prior_path, full_width, full_height, prior, problem_text)) {
        std::cout << "WARNING Depth prior " << prior_path << ": " << problem_text << "; view range used everywhere"
                  << std::endl;
        return;
    }
    const int full_w = prior.cols;
    const int full_h = prior.rows;
    cv::Mat full(full_h, full_w, CV_32FC2, cv::Scalar(0.0f, 0.0f));     // (lo, hi) at full size
    for (int r = 0; r < full_h; ++r) {
        for (int c = 0; c < full_w; ++c) {
            float lo = 0.0f, hi = 0.0f;
            if (prior.type() == CV_32FC2) {
                const cv::Vec2f v = prior.at<cv::Vec2f>(r, c);
                lo = v[0];
                hi = v[1];
            } else {
                const float d = prior.at<float>(r, c);
                if (d > 0.0f) {
                    lo = d - params_host.prior_band_near;
                    hi = d + params_host.prior_band_far;
                }
            }
            if (std::isfinite(lo) && std::isfinite(hi) && lo > 0.0f && hi > lo) {
                full.at<cv::Vec2f>(r, c) = cv::Vec2f(lo, hi);
            }
        }
    }
    // Pool to the working size. APD scales K by s without a half-pixel shift, so working pixel x sits at full-size
    // coordinate x / s and covers [(x - 0.5) / s, (x + 0.5) / s]; the full-size pixels whose centres fall inside that
    // footprint are pooled (lo = min, hi = max over the valid ones), the nearest one when none does. At s = 1 every
    // pixel keeps its own band. A working pixel without any valid full-size pixel stays invalid (view range).
    prior_bounds_host = cv::Mat(height, width, CV_32FC2, cv::Scalar(0.0f, 0.0f));
    const float sx = width / static_cast<float>(full_w);
    const float sy = height / static_cast<float>(full_h);
    auto footprint = [](int x, float s, int n, int &lo_idx, int &hi_idx) {
        lo_idx = std::max(0, (int) std::ceil((x - 0.5f) / s));
        hi_idx = std::min(n - 1, (int) std::floor((x + 0.5f) / s));
        if (hi_idx < lo_idx) {                                   // upsampling: no centre inside, take the nearest
            lo_idx = hi_idx = std::min(n - 1, std::max(0, (int) std::lround(x / s)));
        }
    };
    int valid = 0;
    for (int r = 0; r < height; ++r) {
        int r0, r1;
        footprint(r, sy, full_h, r0, r1);
        for (int c = 0; c < width; ++c) {
            int c0, c1;
            footprint(c, sx, full_w, c0, c1);
            float lo = FLT_MAX, hi = -FLT_MAX;
            for (int rr = r0; rr <= r1; ++rr) {
                for (int cc = c0; cc <= c1; ++cc) {
                    const cv::Vec2f b = full.at<cv::Vec2f>(rr, cc);
                    if (b[1] > b[0]) {
                        lo = std::min(lo, b[0]);
                        hi = std::max(hi, b[1]);
                    }
                }
            }
            if (hi > lo) {
                prior_bounds_host.at<cv::Vec2f>(r, c) = cv::Vec2f(lo, hi);
                valid++;
            }
        }
    }
    std::cout << "Depth prior: " << prior_path.filename().string() << " " << full_w << "x" << full_h << " -> "
              << width << "x" << height << ", valid " << valid << " / " << width * height << std::endl;
    if (valid == 0) {                                          // nothing usable: behave exactly as without a prior
        std::cout << "WARNING Depth prior " << prior_path << " has no valid pixel; view range used everywhere" << std::endl;
        prior_bounds_host.release();
    }
}

void RescaleImageAndCamera(cv::Mat &src, cv::Mat &dst, cv::Mat &depth, Camera &camera) {
    const int cols = depth.cols;
    const int rows = depth.rows;

    if (cols == src.cols && rows == src.rows) {
        dst = src.clone();
        return;
    }

    const float scale_x = cols / static_cast<float>(src.cols);
    const float scale_y = rows / static_cast<float>(src.rows);

    cv::resize(src, dst, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);

    camera.K[0] *= scale_x;
    camera.K[2] *= scale_x;
    camera.K[4] *= scale_y;
    camera.K[5] *= scale_y;
    camera.width = cols;
    camera.height = rows;
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera) {
    float3 pointX;
    float3 tmpX;
    // Reprojection (orthographic: the lateral position does not depend on depth)
    const float scale = (camera.model == CAMERA_ORTHOGRAPHIC ? camera.ref_depth : depth);
    pointX.x = scale * (x - camera.K[2]) / camera.K[0];
    pointX.y = scale * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

void ProjectCamera(const float3 PointX, const Camera &camera, float2 &point, float &depth) {
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    if (camera.model == CAMERA_ORTHOGRAPHIC) {
        depth = tmp.z;
        point.x = camera.K[0] * tmp.x / camera.ref_depth + camera.K[2];
        point.y = camera.K[4] * tmp.y / camera.ref_depth + camera.K[5];
        return;
    }
    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2) {
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle = acosf(dot_product / (cv::norm(v1) * cv::norm(v2)));
    //if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if (angle != angle)
        return 0.0f;

    return angle;
}

// implement a pfm reader
void ReadPFM(const path &filename, cv::Mat &image) {
    ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Error: cannot open file " << filename << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line);
    bool color = false;
    if (line == "Pf") {
        color = false;
    } else if (line == "PF") {
        color = true;
    } else {
        std::cout << "Error: invalid pfm file " << filename << std::endl;
        return;
    }

    std::getline(file, line);
    while (line[0] == '#') {
        std::getline(file, line);
    }

    std::stringstream ss(line);
    int width, height;
    ss >> width >> height;

    std::getline(file, line);
    float scale;
    ss.clear();
    ss.str(line);
    ss >> scale;

    if (color) {
        image = cv::Mat(height, width, CV_32FC3);
        file.read((char *) image.data, sizeof(float) * 3 * width * height);
    } else {
        image = cv::Mat(height, width, CV_32FC1);
        file.read((char *) image.data, sizeof(float) * width * height);
    }

    file.close();

    if (scale < 0) {
        cv::flip(image, image, 0);
    }
}

void WeakVisFilter(
        const std::vector<Problem> &problems,
        const std::vector<Camera> &cameras,
        const std::vector<cv::Mat> &depths,
        const std::vector<cv::Mat> &weaks,
        const std::vector<cv::Mat> &confidences,
        const path &dense_folder,
        std::vector<cv::Mat> &skip_weaks,
        const WeakFilterParams &wf = WeakFilterParams()
) {
    const int num_images = cameras.size();
    // confidence at (r, c) as stored: APD writes CV_8UC1 counts and LoadFusionViews admits no other type.
    // Upstream read the 8-bit map with at<float>, i.e. it compared reinterpreted bytes and read past the end of
    // the matrix in its last rows.
    const auto confidence_at = [](const cv::Mat &m, int r, int c) {
        return m.at<uchar>(r, c);
    };
    const auto task = [&](int ref_index) {
        if (depths[ref_index].empty())
            return;                                          // view without usable depth maps (no skip.png)
        const int width = depths[ref_index].cols;
        const int height = depths[ref_index].rows;
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                if (weaks[ref_index].at<uchar>(r, c) == WEAK) {
                    float ref_depth = depths[ref_index].at<float>(r, c);
                    float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
                    int strong_occluded = 0;
                    int weak_occluded = 0;
                    for (int src_index = 0; src_index < num_images; ++src_index) {
                        const auto &ref_cam = cameras[ref_index];
                        const auto &src_cam = cameras[src_index];
                        if (ref_index == src_index || depths[src_index].empty())
                            continue;                        // itself, or a view without usable depth maps
                        cv::Vec3f a(ref_cam.c[0] - PointX.x, ref_cam.c[1] - PointX.y, ref_cam.c[2] - PointX.z);
                        cv::Vec3f b(src_cam.c[0] - PointX.x, src_cam.c[1] - PointX.y, src_cam.c[2] - PointX.z);
                        float angle = GetAngle(a, b);
                        angle = angle * 180.0f / M_PI;  // convert to degree
                        if (angle > wf.max_view_angle_deg) {
                            continue;
                        }
                        float2 point;
                        float proj_depth;
                        ProjectCamera(PointX, src_cam, point, proj_depth);
                        if (proj_depth <= 0.0f)
                            continue;
                        int src_r = int(point.y + 0.5f);
                        int src_c = int(point.x + 0.5f);
                        const int src_cols = depths[src_index].cols;
                        const int src_rows = depths[src_index].rows;
                        if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                            float src_depth = depths[src_index].at<float>(src_r, src_c);
                            // margin by which the point must lie in front of the source surface
                            const float margin = (wf.depth_mode == DEPTH_TEST_ABSOLUTE ? wf.depth_abs
                                                                                       : wf.depth_rel * src_depth);
                            if (weaks[src_index].at<uchar>(src_r, src_c) == STRONG) {
                                if (proj_depth < src_depth - margin) {
                                    strong_occluded++;
                                }
                            } else if (weaks[src_index].at<uchar>(src_r, src_c) == WEAK) {
                                const bool src_less_confident = confidence_at(confidences[src_index], src_r, src_c) <
                                                                confidence_at(confidences[ref_index], r, c);
                                if (src_less_confident) {
                                    if (proj_depth < src_depth - margin) {
                                        weak_occluded++;
                                    }
                                }
                            }
                        }
                    }
                    if (strong_occluded >= wf.strong_occluded_min || weak_occluded >= wf.weak_occluded_min) {
                        skip_weaks[ref_index].at<uchar>(r, c) = 1;
                        continue;
                    }
                }
            }
        }
        cv::Mat skip_weak_img = cv::Mat::zeros(height, width, CV_8UC1);
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                if (skip_weaks[ref_index].at<uchar>(r, c) == 1) {
                    skip_weak_img.at<uchar>(r, c) = 255;
                }
            }
        }
        int ref_image_id = problems[ref_index].ref_image_id;
        cv::imwrite((dense_folder / path("APD") / path(ToFormatIndex(ref_image_id)) / path("skip.png")).string(),
                    skip_weak_img);
        printf("filter for image %d done\n", ref_image_id);
    };
    // get max threads num
    const int num_threads = MIN(std::thread::hardware_concurrency(), depths.size());
    ThreadPool pool(num_threads);
    std::vector<std::future<void>> results;
    for (int ref_index = 0; ref_index < num_images; ++ref_index) {
        results.emplace_back(pool.enqueue(task, ref_index));
    }
    for (auto &&result: results) {
        result.get();
    }
}

float OpticalAxisAngleDeg(const Camera &a, const Camera &b) {
    // The third row of R is the camera's viewing direction expressed in world coordinates.
    double dot = double(a.R[6]) * b.R[6] + double(a.R[7]) * b.R[7] + double(a.R[8]) * b.R[8];
    dot = std::max(-1.0, std::min(1.0, dot));  // guard acos against rounding just outside [-1, 1]
    return static_cast<float>(std::acos(dot) * 180.0 / M_PI);
}

bool ViewAngleAllowed(const Camera &ref, const Camera &src, float min_deg, float max_deg) {
    if (min_deg < 0.0f && max_deg < 0.0f) {
        return true;  // window switched off
    }
    const float angle = OpticalAxisAngleDeg(ref, src);
    if (min_deg >= 0.0f && angle < min_deg) {
        return false;
    }
    if (max_deg >= 0.0f && angle > max_deg) {
        return false;
    }
    return true;
}

// Per-view inputs of the three fusion variants, in problem order: slot i always belongs to problems[i], and
// imageIdToindexMap maps each reference id to the slot of its first occurrence. A view whose result files are
// missing, of the wrong type or of inconsistent sizes, whose image or cam file cannot be read, or whose reference id
// repeats an earlier one keeps its slot with empty matrices (depth 0 x 0), so it takes part in nothing but every
// index stays in range. Upstream skipped such a view after mapping its id, which shortened the vectors and shifted
// every later view by one slot.
struct FusionViews {
    std::vector<cv::Mat> images;
    std::vector<Camera> cameras;
    std::vector<cv::Mat> depths;
    std::vector<cv::Mat> normals;
    std::vector<cv::Mat> masks;
    std::vector<cv::Mat> weaks;
    std::vector<cv::Mat> confidences;
    std::vector<cv::Mat> skip_weaks;
    std::unordered_map<int, int> imageIdToindexMap;
};

static void LoadFusionViews(const path &dense_folder, const std::vector<Problem> &problems, FusionViews &views) {
    const int num_images = static_cast<int>(problems.size());
    const path image_folder = dense_folder / path("images");
    const path cam_folder = dense_folder / path("cams");
    for (int i = 0; i < num_images; ++i) {
        const auto &problem = problems[i];
        std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + problem.img_ext);
        if (!views.imageIdToindexMap.emplace(problem.ref_image_id, i).second) {
            // pair.txt lists this reference id again: fusion of this problem uses the first occurrence's slot (as
            // upstream did), so its own slot stays empty instead of holding a second copy of the same maps
            std::cout << "WARNING: image " << i << " repeats reference id " << problem.ref_image_id
                      << "; it is fused from the maps of image " << views.imageIdToindexMap.at(problem.ref_image_id)
                      << std::endl;
            views.images.emplace_back();
            views.cameras.emplace_back(Camera{});
            views.depths.emplace_back();
            views.normals.emplace_back();
            views.masks.emplace_back();
            views.weaks.emplace_back();
            views.skip_weaks.emplace_back();
            views.confidences.emplace_back();
            continue;
        }
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
        Camera camera{};                                     // zeros if the file cannot be read
        const bool camera_ok = ReadCamera(cam_path, camera);

        path depth_path = problem.result_folder / path("depths.bin");
        path normal_path = problem.result_folder / path("normals.bin");
        path weak_path = problem.result_folder / path("weak.bin");
        path confidence_path = problem.result_folder / path("confidence.bin");
        cv::Mat depth, normal, weak, confidence;
        ReadBinMat(depth_path, depth);
        ReadBinMat(normal_path, normal);
        ReadBinMat(weak_path, weak);
        ReadBinMat(confidence_path, confidence);
        // check the size (and the element types the fusion reads with at<float> / at<cv::Vec3f>)
        bool usable = true;
        if (!camera_ok) {
            std::cout << "Error: cannot read " << cam_path << std::endl;
            usable = false;
        } else if (depth.empty() || depth.type() != CV_32FC1) {
            std::cout << "Error: depth map missing or not CV_32FC1" << std::endl;
            usable = false;
        } else if (normal.cols != depth.cols || normal.rows != depth.rows) {
            std::cout << "Error: normal size is not equal to depth size" << std::endl;
            usable = false;
        } else if (normal.type() != CV_32FC3) {
            std::cout << "Error: normal map is not CV_32FC3" << std::endl;
            usable = false;
        } else if (weak.cols != depth.cols || weak.rows != depth.rows) {
            std::cout << "Error: weak size is not equal to depth size" << std::endl;
            usable = false;
        } else if (weak.type() != CV_8UC1) {
            std::cout << "Error: weak map is not CV_8UC1" << std::endl;
            usable = false;
        } else if (confidence.cols != depth.cols || confidence.rows != depth.rows) {
            std::cout << "Error: confidence size is not equal to depth size" << std::endl;
            usable = false;
        } else if (confidence.type() != CV_8UC1) {
            std::cout << "Error: confidence map is not CV_8UC1" << std::endl;
            usable = false;
        } else if (image.empty()) {
            std::cout << "Error: cannot read " << image_path << std::endl;
            usable = false;
        }
        if (!usable) {
            std::cout << "WARNING: image " << i << " (id " << problem.ref_image_id << ") takes no part in fusion"
                      << std::endl;
            views.images.emplace_back();
            views.cameras.emplace_back(camera);              // never used for a pixel: its depth map is 0 x 0
            views.depths.emplace_back();
            views.normals.emplace_back();
            views.masks.emplace_back();
            views.weaks.emplace_back();
            views.skip_weaks.emplace_back();
            views.confidences.emplace_back();
            continue;
        }
        cv::Mat scaled_image;
        RescaleImageAndCamera(image, scaled_image, depth, camera);
        views.images.emplace_back(scaled_image);
        views.cameras.emplace_back(camera);
        views.depths.emplace_back(depth);
        views.normals.emplace_back(normal);
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        views.masks.emplace_back(mask);
        views.weaks.emplace_back(weak);
        cv::Mat skip_weak = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        views.skip_weaks.emplace_back(skip_weak);
        views.confidences.emplace_back(confidence);
    }
}

// Slot of every source view of every problem (-1 for an id that is not a reference view of this run, with a
// warning). Upstream looked ids up with operator[], which silently mapped an unknown id to slot 0 (and may insert,
// so it must not be called from worker threads).
static std::vector<std::vector<int>> SourceSlots(const std::vector<Problem> &problems,
                                                 const std::unordered_map<int, int> &imageIdToindexMap) {
    std::vector<std::vector<int>> slots(problems.size());
    for (size_t i = 0; i < problems.size(); ++i) {
        for (const int src_id: problems[i].src_image_ids) {
            const auto it = imageIdToindexMap.find(src_id);
            if (it == imageIdToindexMap.end()) {
                std::cout << "WARNING: source id " << src_id << " of image " << i << " (id "
                          << problems[i].ref_image_id << ") is not a reference view of this run; skipped" << std::endl;
                slots[i].push_back(-1);
            } else {
                slots[i].push_back(it->second);
            }
        }
    }
    return slots;
}

void RunFusion(
    const path &dense_folder,
    const std::vector<Problem> &problems,
    const std::string &name,
    bool weak_filter,
    bool export_color,
    const FusionParams &fp,
    const WeakFilterParams &wf
) {
    int num_images = problems.size();

    // to avoid out of memory, we release the gray image cache!
    if (memory_cache != nullptr) {
        memory_cache->img_cache.clear();
    }

    FusionViews views;
    LoadFusionViews(dense_folder, problems, views);
    std::vector<cv::Mat> &images = views.images;
    std::vector<Camera> &cameras = views.cameras;
    std::vector<cv::Mat> &depths = views.depths;
    std::vector<cv::Mat> &normals = views.normals;
    std::vector<cv::Mat> &masks = views.masks;
    std::vector<cv::Mat> &weaks = views.weaks;
    std::vector<cv::Mat> &confidences = views.confidences;
    std::vector<cv::Mat> &skip_weaks = views.skip_weaks;
    const std::unordered_map<int, int> &imageIdToindexMap = views.imageIdToindexMap;
    const std::vector<std::vector<int>> src_slots = SourceSlots(problems, imageIdToindexMap);

    if (weak_filter) {
        WeakVisFilter(problems, cameras, depths, weaks, confidences, dense_folder, skip_weaks, wf);
    }

    // Silhouette-edge trim (fusion.silhouette_trim_px): pixels within the band on either side of each view's own
    // silhouette edge are marked as used, so they are neither references nor sources.
    if (fp.silhouette_trim_px > 0) {
        for (int i = 0; i < num_images; ++i) {
            const int idx = imageIdToindexMap.at(problems[i].ref_image_id);
            if (depths[idx].empty())
                continue;                                            // view without usable depth maps
            path sa_path = dense_folder / path("sa_masks") / path(ToFormatIndex(problems[i].ref_image_id) + ".bin");
            cv::Mat sa;
            if (!exists(sa_path) || !ReadBinMat(sa_path, sa)) {
                std::cout << "silhouette_trim_px: no silhouette for image " << i << " (" << sa_path << "), not trimmed"
                          << std::endl;
                continue;
            }
            if (sa.empty() || sa.type() != CV_8UC1) {                // non-empty one-channel 8-bit object mask expected
                std::cout << "WARNING silhouette_trim_px: " << sa_path << " is not a 1-channel 8-bit mask (type "
                          << sa.type() << "), not trimmed" << std::endl;
                continue;
            }
            cv::Mat inside;
            cv::resize(sa != 0, inside, cv::Size(depths[idx].cols, depths[idx].rows), 0, 0, cv::INTER_NEAREST);
            cv::Mat outside = (inside == 0);
            cv::Mat d_in, d_out;
            cv::distanceTransform(inside, d_in, cv::DIST_L2, 5);     // inside pixels: distance to the background
            cv::distanceTransform(outside, d_out, cv::DIST_L2, 5);   // outside pixels: distance to the object
            int trimmed = 0;
            for (int r = 0; r < inside.rows; ++r) {
                for (int c = 0; c < inside.cols; ++c) {
                    const float d = inside.at<uchar>(r, c) ? d_in.at<float>(r, c) : d_out.at<float>(r, c);
                    if (d <= (float) fp.silhouette_trim_px) {
                        masks[idx].at<uchar>(r, c) = 1;
                        ++trimmed;
                    }
                }
            }
            std::cout << "silhouette_trim_px " << fp.silhouette_trim_px << ": image " << i << " trimmed " << trimmed
                      << " pixels" << std::endl;
        }
    }
    const bool use_incident_max = fp.incident_max_deg >= 0.0f;
    const bool use_incident_prior = fp.incident_sigma_deg > 0.0f;
    const float incident_cos_min = use_incident_max ? cosf(fp.incident_max_deg * (float) M_PI / 180.0f) : -2.0f;
    const float incident_two_sigma2 = use_incident_prior ?
        2.0f * powf(fp.incident_sigma_deg * (float) M_PI / 180.0f, 2.0f) : 1.0f;
    // cosine of the angle between a pixel normal (world frame) and the direction from its 3D point to its camera;
    // an orthographic camera looks along its optical axis everywhere, so the direction is minus the axis (row 3 of R)
    auto incident_cos = [](const cv::Vec3f &n, const float3 &X, const Camera &cam) {
        cv::Vec3f v = (cam.model == CAMERA_ORTHOGRAPHIC) ? cv::Vec3f(-cam.R[6], -cam.R[7], -cam.R[8])
                                                         : cv::Vec3f(cam.c[0] - X.x, cam.c[1] - X.y, cam.c[2] - X.z);
        const float nn = (float) cv::norm(n), vn = (float) cv::norm(v);
        return (nn > 0.0f && vn > 0.0f) ? (float) n.dot(v) / (nn * vn) : 1.0f;
    };
    auto incident_prior = [&](float cosk) {
        const float k = acosf(std::max(-1.0f, std::min(1.0f, cosk)));
        return expf(-k * k / incident_two_sigma2);
    };

    const bool use_angle_window = (fp.view_min_angle_deg >= 0.0f || fp.view_max_angle_deg >= 0.0f);

    // ---- behind-surface test (fusion.occlusion_test): per-view support maps before fusion ----
    // support[v](r, c) = number of agreeing sources of pixel (r, c) of view v when the fusion below would accept that
    // pixel as a reference, else 0. It applies fusion's own rules: trimmed (masks) and see-through (skip_weaks)
    // reference pixels, trimmed source pixels, sources outside the optical-axis angle window and unknown source ids
    // take no part; reprojection, depth (fusion.depth_mode), normal and incident-angle tests, the consistency score
    // and min_consistent. Unlike fusion, no pixel is consumed (with mask_used_pixels, pixels that fusion consumes
    // keep their support), so the map is independent of the fusion order. A repeated reference id has one map,
    // built from the source list of its first occurrence.
    std::vector<cv::Mat> support;
    long long occlusion_tested = 0, occlusion_rejected = 0;
    if (fp.occlusion_test) {
        support.assign(num_images, cv::Mat());
        auto support_task = [&](int i) {
            const auto &problem = problems[i];
            const int v = imageIdToindexMap.at(problem.ref_image_id);
            std::vector<int> srcs;                           // the sources fusion would use for this reference
            for (const int u: src_slots[i]) {
                if (u < 0 || depths[u].empty())
                    continue;                                // unknown source id or view without usable depth maps
                if (use_angle_window && !ViewAngleAllowed(cameras[v], cameras[u], fp.view_min_angle_deg,
                                                          fp.view_max_angle_deg))
                    continue;                                // outside the optical-axis angle window
                srcs.push_back(u);
            }
            cv::Mat s = cv::Mat::zeros(depths[v].rows, depths[v].cols, CV_8UC1);
            for (int r = 0; r < depths[v].rows; ++r) {
                for (int c = 0; c < depths[v].cols; ++c) {
                    if (masks[v].at<uchar>(r, c) == 1 || skip_weaks[v].at<uchar>(r, c) == 1)
                        continue;                            // trimmed or see-through: never a fusion reference
                    const float d = depths[v].at<float>(r, c);
                    if (d <= 0.0f)
                        continue;
                    const cv::Vec3f n = normals[v].at<cv::Vec3f>(r, c);
                    const float3 X = Get3DPointonWorld(c, r, d, cameras[v]);
                    float ref_prior = 1.0f;
                    if (use_incident_max || use_incident_prior) {
                        const float ref_cos = incident_cos(n, X, cameras[v]);
                        if (use_incident_max && ref_cos < incident_cos_min)
                            continue;
                        if (use_incident_prior)
                            ref_prior = incident_prior(ref_cos);
                    }
                    int count = 0;
                    float consistency = 0.0f;
                    for (const int u: srcs) {
                        float2 pt;
                        float pd;
                        ProjectCamera(X, cameras[u], pt, pd);
                        const int sr = int(pt.y + 0.5f), sc = int(pt.x + 0.5f);
                        if (sc < 0 || sc >= depths[u].cols || sr < 0 || sr >= depths[u].rows)
                            continue;
                        if (masks[u].at<uchar>(sr, sc) == 1)
                            continue;                        // trimmed source pixel
                        const float sd = depths[u].at<float>(sr, sc);
                        if (sd <= 0.0f)
                            continue;
                        const cv::Vec3f sn = normals[u].at<cv::Vec3f>(sr, sc);
                        const float3 Y = Get3DPointonWorld(sc, sr, sd, cameras[u]);
                        float2 back;
                        float bd;
                        ProjectCamera(Y, cameras[v], back, bd);
                        const float reproj = sqrt(pow(c - back.x, 2) + pow(r - back.y, 2));
                        float src_prior = 1.0f;
                        if (use_incident_max || use_incident_prior) {
                            const float src_cos = incident_cos(sn, Y, cameras[u]);
                            if (use_incident_max && src_cos < incident_cos_min)
                                continue;
                            if (use_incident_prior)
                                src_prior = incident_prior(src_cos);
                        }
                        const float angle = fp.normal_test ? GetAngle(n, sn) : 0.0f;
                        bool depth_ok = true;                // same test and score term as the fusion loop
                        float depth_score = 0.0f;
                        if (fp.depth_mode == DEPTH_TEST_RELATIVE) {
                            const float rel = fabs(bd - d) / d;
                            depth_ok = rel < fp.depth_rel;
                            depth_score = fp.score_w_depth * rel;
                        } else if (fp.depth_mode == DEPTH_TEST_ABSOLUTE) {
                            const float abs_diff = fabs(bd - d);
                            depth_ok = abs_diff < fp.depth_abs;
                            depth_score = fp.score_w_depth_scaled * abs_diff / fp.depth_abs;
                        } else if (fp.depth_mode == DEPTH_TEST_PIXEL) {
                            float3 shifted = Get3DPointonWorld(c, r, bd, cameras[v]);
                            float2 sp;
                            float sdd;
                            ProjectCamera(shifted, cameras[u], sp, sdd);
                            const float px_shift = sqrt(pow(pt.x - sp.x, 2) + pow(pt.y - sp.y, 2));
                            depth_ok = px_shift < fp.depth_px;
                            depth_score = fp.score_w_depth_scaled * px_shift / fp.depth_px;
                        }
                        if (reproj < fp.reproj_px && depth_ok && angle < fp.normal_max_rad) {
                            ++count;
                            const float tmp_index = fp.score_w_reproj * reproj + depth_score + angle * fp.score_w_normal;
                            consistency += exp(-tmp_index) * src_prior;
                        }
                    }
                    const float factor = (weaks[v].at<uchar>(r, c) == WEAK ? fp.score_thresh_weak
                                                                           : fp.score_thresh_strong);
                    if (count >= fp.min_consistent && ref_prior * consistency > factor * count)
                        s.at<uchar>(r, c) = (uchar) std::min(count, 255);
                }
            }
            support[v] = s;
        };
        const int num_threads = std::max(1, (int) std::min(std::thread::hardware_concurrency(), (unsigned) num_images));
        ThreadPool pool(num_threads);
        std::vector<std::future<void>> jobs;
        for (int i = 0; i < num_images; ++i) {
            if (imageIdToindexMap.at(problems[i].ref_image_id) != i)
                continue;                                    // duplicated reference id: its slot has a job already
            jobs.emplace_back(pool.enqueue(support_task, i));
        }
        for (auto &&job: jobs) {
            job.get();
        }
        std::cout << "occlusion_test: support maps done" << std::endl;
    }
    // True when another view k sees, at the pixel where X projects, an occluding surface:
    // - k lies inside the reference's optical-axis angle window (view_allowed) and both X's normal and the
    //   occluder's normal face k (cos >= occlusion_min_cos);
    // - the occluder is supported: support >= occlusion_min_support (0 for trimmed and see-through pixels);
    // - it lies in front of X by more than occlusion_tol, measured as z-depth in view k (pd - kd);
    // - fusion's own reprojection and depth tests (fusion.reproj_px, fusion.depth_mode) for the pair
    //   (reference, k) reject it, i.e. fusion would not treat it and X as one surface. This keeps pixel rounding
    //   and depth noise on slanted surfaces from counting as "in front".
    auto behind_surface = [&](const float3 &X, const cv::Vec3f &n, int ref_index, int r, int c, float ref_depth,
                              const std::vector<char> &view_allowed) {
        for (int k = 0; k < num_images; ++k) {
            if (k == ref_index || support[k].empty() || !view_allowed[k])
                continue;
            const Camera &cam = cameras[k];
            if (incident_cos(n, X, cam) < fp.occlusion_min_cos)
                continue;                                     // X faces away from view k: k cannot see it anyway
            float2 pt;
            float pd;
            ProjectCamera(X, cam, pt, pd);
            if (pd <= 0.0f)
                continue;
            const int kr = int(pt.y + 0.5f), kc = int(pt.x + 0.5f);
            if (kc < 0 || kc >= depths[k].cols || kr < 0 || kr >= depths[k].rows)
                continue;
            const float kd = depths[k].at<float>(kr, kc);
            if (kd <= 0.0f || pd - kd <= fp.occlusion_tol)
                continue;                                     // nothing in front of X in view k
            if (support[k].at<uchar>(kr, kc) < fp.occlusion_min_support)
                continue;                                     // the front surface is not better supported
            const float3 Y = Get3DPointonWorld(kc, kr, kd, cam);
            const cv::Vec3f kn = normals[k].at<cv::Vec3f>(kr, kc);
            if (incident_cos(kn, Y, cam) < fp.occlusion_min_cos)
                continue;                                     // grazing occluder (limb flare): not trusted
            float2 back;
            float bd;
            ProjectCamera(Y, cameras[ref_index], back, bd);
            const float reproj = sqrt(pow(c - back.x, 2) + pow(r - back.y, 2));
            bool depth_ok = true;                             // DEPTH_TEST_OFF: reprojection alone decides
            if (fp.depth_mode == DEPTH_TEST_RELATIVE) {
                depth_ok = fabs(bd - ref_depth) / ref_depth < fp.depth_rel;
            } else if (fp.depth_mode == DEPTH_TEST_ABSOLUTE) {
                depth_ok = fabs(bd - ref_depth) < fp.depth_abs;
            } else if (fp.depth_mode == DEPTH_TEST_PIXEL) {
                float3 shifted = Get3DPointonWorld(c, r, bd, cameras[ref_index]);
                float2 sp;
                float sdd;
                ProjectCamera(shifted, cam, sp, sdd);
                depth_ok = sqrt(pow(pt.x - sp.x, 2) + pow(pt.y - sp.y, 2)) < fp.depth_px;
            }
            if (reproj < fp.reproj_px && depth_ok)
                continue;                                     // fusion would call it the same surface as X
            return true;
        }
        return false;
    };

    std::vector<PointList> PointCloud;
    PointCloud.clear();

    for (int i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        const auto &problem = problems[i];
        int ref_index = imageIdToindexMap.at(problem.ref_image_id);
        const int cols = depths[ref_index].cols;
        const int rows = depths[ref_index].rows;
        int num_ngb = problem.src_image_ids.size();
        // optional optical-axis angle window: decided once per (reference, source) pair
        // (an unknown source id or a view without usable depth maps is never allowed)
        std::vector<char> src_allowed(num_ngb, 1);
        for (int j = 0; j < num_ngb; ++j) {
            const int src_index = src_slots[i][j];
            if (src_index < 0 || depths[src_index].empty()) {
                src_allowed[j] = 0;
            } else if (use_angle_window) {
                src_allowed[j] = ViewAngleAllowed(cameras[ref_index], cameras[src_index], fp.view_min_angle_deg,
                                                  fp.view_max_angle_deg) ? 1 : 0;
            }
        }
        // occlusion_test: views that may act as occluders of this reference (the same angle window, over all views)
        std::vector<char> occluder_allowed;
        if (fp.occlusion_test) {
            occluder_allowed.assign(num_images, 1);
            if (use_angle_window) {
                for (int k = 0; k < num_images; ++k) {
                    if (depths[k].empty()) {
                        occluder_allowed[k] = 0;             // view without usable depth maps
                        continue;
                    }
                    occluder_allowed[k] = ViewAngleAllowed(cameras[ref_index], cameras[k], fp.view_min_angle_deg,
                                                           fp.view_max_angle_deg) ? 1 : 0;
                }
            }
        }
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (masks[ref_index].at<uchar>(r, c) == 1) {
                    continue;
                }
                if (skip_weaks[ref_index].at<uchar>(r, c) == 1) {
                    continue;
                }

                float ref_depth = depths[ref_index].at<float>(r, c);
                if (ref_depth <= 0.0)
                    continue;
                const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
                float3 consistent_Point = PointX;
                float ref_prior = 1.0f;
                if (use_incident_max || use_incident_prior) {
                    const float ref_cos = incident_cos(ref_normal, PointX, cameras[ref_index]);
                    if (use_incident_max && ref_cos < incident_cos_min)
                        continue;                            // grazing reference pixel
                    if (use_incident_prior)
                        ref_prior = incident_prior(ref_cos);
                }

                int num_consistent = 0;
                float dynamic_consistency = 0.0f;
                std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
                for (int j = 0; j < num_ngb; ++j) {
                    if (!src_allowed[j])
                        continue;
                    int src_index = src_slots[i][j];
                    const int src_cols = depths[src_index].cols;
                    const int src_rows = depths[src_index].rows;
                    float2 point;
                    float proj_depth;
                    ProjectCamera(PointX, cameras[src_index], point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (masks[src_index].at<uchar>(src_r, src_c) == 1)
                            continue;
                        float src_depth = depths[src_index].at<float>(src_r, src_c);
                        if (src_depth <= 0.0)
                            continue;
                        const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
                        float2 tmp_pt;
                        ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float src_prior = 1.0f;
                        if (use_incident_max || use_incident_prior) {
                            const float src_cos = incident_cos(src_normal, tmp_X, cameras[src_index]);
                            if (use_incident_max && src_cos < incident_cos_min)
                                continue;                    // grazing source pixel
                            if (use_incident_prior)
                                src_prior = incident_prior(src_cos);
                        }
                        // normals.bin holds world-frame normals (GetDepthandNormal rotates them), so this
                        // comparison is valid for any angle between the two cameras
                        float angle = fp.normal_test ? GetAngle(ref_normal, src_normal) : 0.0f;

                        // depth agreement: depth_ok gates the match, depth_score enters the consistency score
                        bool depth_ok = true;
                        float depth_score = 0.0f;
                        if (fp.depth_mode == DEPTH_TEST_RELATIVE) {
                            depth_ok = relative_depth_diff < fp.depth_rel;
                            depth_score = fp.score_w_depth * relative_depth_diff;
                        } else if (fp.depth_mode == DEPTH_TEST_ABSOLUTE) {
                            const float abs_diff = fabs(proj_depth - ref_depth);  // world units along the ref axis
                            depth_ok = abs_diff < fp.depth_abs;
                            depth_score = fp.score_w_depth_scaled * abs_diff / fp.depth_abs;
                        } else if (fp.depth_mode == DEPTH_TEST_PIXEL) {
                            // move the reference pixel to the depth the source view proposes and measure how
                            // far its projection into the source image shifts
                            float3 shifted_X = Get3DPointonWorld(c, r, proj_depth, cameras[ref_index]);
                            float2 shifted_pt;
                            float shifted_depth;
                            ProjectCamera(shifted_X, cameras[src_index], shifted_pt, shifted_depth);
                            const float px_shift = sqrt(pow(point.x - shifted_pt.x, 2) +
                                                        pow(point.y - shifted_pt.y, 2));
                            depth_ok = px_shift < fp.depth_px;
                            depth_score = fp.score_w_depth_scaled * px_shift / fp.depth_px;
                        }

                        if (reproj_error < fp.reproj_px && depth_ok && angle < fp.normal_max_rad) {
                            used_list[j].x = src_c;
                            used_list[j].y = src_r;
                            float tmp_index = fp.score_w_reproj * reproj_error + depth_score +
                                              angle * fp.score_w_normal;
                            dynamic_consistency += exp(-tmp_index) * src_prior;
                            num_consistent++;
                            if (fp.average_position) {
                                consistent_Point.x += tmp_X.x;
                                consistent_Point.y += tmp_X.y;
                                consistent_Point.z += tmp_X.z;
                            }
                        }
                    }
                }
                float factor = (weaks[ref_index].at<uchar>(r, c) == WEAK ? fp.score_thresh_weak
                                                                         : fp.score_thresh_strong);
                if (num_consistent >= fp.min_consistent && (ref_prior * dynamic_consistency > factor * num_consistent)) {
                    if (fp.occlusion_test && num_consistent <= fp.occlusion_max_support) {
                        ++occlusion_tested;
                        if (behind_surface(PointX, ref_normal, ref_index, r, c, ref_depth, occluder_allowed)) {
                            ++occlusion_rejected;            // weakly supported point under a better-supported surface
                            continue;
                        }
                    }
                    if (fp.average_position) {
                        consistent_Point.x /= (num_consistent + 1);
                        consistent_Point.y /= (num_consistent + 1);
                        consistent_Point.z /= (num_consistent + 1);
                    }
                    PointList point3D;
                    point3D.coord = consistent_Point;
                    float consistent_Color[3] = {(float) images[ref_index].at<cv::Vec3b>(r, c)[0],
                                                 (float) images[ref_index].at<cv::Vec3b>(r, c)[1],
                                                 (float) images[ref_index].at<cv::Vec3b>(r, c)[2]};
                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        int src_index = src_slots[i][j];      // used_list is only set for known sources
                        if (fp.mask_used_pixels) {
                            masks[src_index].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                        }
                        const auto &color = images[src_index].at<cv::Vec3b>(used_list[j].y, used_list[j].x);
                        consistent_Color[0] += color[0];
                        consistent_Color[1] += color[1];
                        consistent_Color[2] += color[2];
                    }
                    consistent_Color[0] /= (num_consistent + 1);
                    consistent_Color[1] /= (num_consistent + 1);
                    consistent_Color[2] /= (num_consistent + 1);
                    point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.emplace_back(point3D);
                }

            }
        }
    }
    if (fp.occlusion_test) {
        std::cout << "occlusion_test: tested " << occlusion_tested << " weakly supported points, rejected "
                  << occlusion_rejected << std::endl;
    }
    path ply_path = dense_folder / path("APD") / path(name);
    ExportPointCloud(ply_path, PointCloud, export_color);
}

void RunFusion_TAT_I(
    const path &dense_folder,
    const std::vector<Problem> &problems,
    const std::string &name,
    bool weak_filter,
    bool export_color
) {
    int num_images = problems.size();
    const float dist_base = 0.25f;
    const float depth_base = 1.0f / 3500.0f;

    const float angle_base = 0.06981317007977318f; // 4 degree
    const float angle_grad = 0.05235987755982988f; // 3 degree

    // to avoid out of memory, we release the gray image cache!
    if (memory_cache != nullptr) {
        memory_cache->img_cache.clear();
    }

    FusionViews views;
    LoadFusionViews(dense_folder, problems, views);
    std::vector<cv::Mat> &images = views.images;
    std::vector<Camera> &cameras = views.cameras;
    std::vector<cv::Mat> &depths = views.depths;
    std::vector<cv::Mat> &normals = views.normals;
    std::vector<cv::Mat> &masks = views.masks;
    std::vector<cv::Mat> &weaks = views.weaks;
    std::vector<cv::Mat> &confidences = views.confidences;
    std::vector<cv::Mat> &skip_weaks = views.skip_weaks;
    const std::unordered_map<int, int> &imageIdToindexMap = views.imageIdToindexMap;
    const std::vector<std::vector<int>> src_slots = SourceSlots(problems, imageIdToindexMap);

    if (weak_filter) {
        WeakVisFilter(problems, cameras, depths, weaks, confidences, dense_folder, skip_weaks);
    }

    std::vector<PointList> PointCloud;
    PointCloud.clear();

    struct CostData {
        float dist;
        float depth;
        float angle;
        int src_r = -1;
        int src_c = -1;
        bool use = false;

        CostData() {
            dist = FLT_MAX;
            depth = FLT_MAX;
            angle = FLT_MAX;
        }
    };


    for (int i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        const auto &problem = problems[i];
        int ref_index = imageIdToindexMap.at(problem.ref_image_id);
        const int cols = depths[ref_index].cols;
        const int rows = depths[ref_index].rows;
        int num_ngb = problem.src_image_ids.size();
        std::vector<CostData> diff(num_ngb, CostData());
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (skip_weaks[ref_index].at<uchar>(r, c) == 1) {
                    continue;
                }
                float ref_depth = depths[ref_index].at<float>(r, c);
                if (ref_depth <= 0.0)
                    continue;
                const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
                // values of this pixel only (upstream kept a source's values from an earlier pixel when the source
                // was skipped here, so a stale match could still count)
                std::fill(diff.begin(), diff.end(), CostData());

                float3 consistent_Point = PointX;
                for (int j = 0; j < num_ngb; ++j) {
                    int src_index = src_slots[i][j];
                    if (src_index < 0 || depths[src_index].empty())
                        continue;                            // unknown or unusable source: diff[j] stays unset
                    const int src_cols = depths[src_index].cols;
                    const int src_rows = depths[src_index].rows;
                    float2 point;
                    float proj_depth;
                    ProjectCamera(PointX, cameras[src_index], point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (masks[src_index].at<uchar>(src_r, src_c) == 1)
                            continue;
                        float src_depth = depths[src_index].at<float>(src_r, src_c);
                        if (src_depth <= 0.0)
                            continue;
                        const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
                        float2 tmp_pt;
                        ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);
                        diff[j].dist = reproj_error;
                        diff[j].depth = relative_depth_diff;
                        diff[j].angle = angle;
                        diff[j].src_r = src_r;
                        diff[j].src_c = src_c;
                    }
                }
                for (int k = 2; k <= num_ngb; ++k) {
                    int count = 0;
                    for (int j = 0; j < num_ngb; ++j) {
                        diff[j].use = false;
                        if (diff[j].dist < k * dist_base && diff[j].depth < k * depth_base &&
                            diff[j].angle < (k * angle_grad + angle_base)) {
                            count++;
                            diff[j].use = true;
                        }
                    }
                    if (count >= k) {
                        PointList point3D;
                        float consistent_Color[3] = {(float) images[ref_index].at<cv::Vec3b>(r, c)[0],
                                                     (float) images[ref_index].at<cv::Vec3b>(r, c)[1],
                                                     (float) images[ref_index].at<cv::Vec3b>(r, c)[2]};
                        for (int j = 0; j < num_ngb; ++j) {
                            if (diff[j].use) {
                                int src_index = src_slots[i][j];
                                consistent_Color[0] += (float) images[src_index].at<cv::Vec3b>(diff[j].src_r,
                                                                                               diff[j].src_c)[0];
                                consistent_Color[1] += (float) images[src_index].at<cv::Vec3b>(diff[j].src_r,
                                                                                               diff[j].src_c)[1];
                                consistent_Color[2] += (float) images[src_index].at<cv::Vec3b>(diff[j].src_r,
                                                                                               diff[j].src_c)[2];
                            }
                        }
                        consistent_Color[0] /= (count + 1.0f);
                        consistent_Color[1] /= (count + 1.0f);
                        consistent_Color[2] /= (count + 1.0f);

                        point3D.coord = consistent_Point;
                        point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                        PointCloud.emplace_back(point3D);
                        masks[ref_index].at<uchar>(r, c) = 1;
                        break;
                    }
                }
            }
        }
    }
    path ply_path = dense_folder / path("APD") / path(name);
    ExportPointCloud(ply_path, PointCloud, export_color);
}

void RunFusion_TAT_A(
    const path &dense_folder,
    const std::vector<Problem> &problems,
    const std::string &name,
    bool weak_filter,
    bool export_color
) {
    int num_images = problems.size();
    const float dist_base = 0.25f;
    const float depth_base = 1.0f / 3000.0f;

    // to avoid out of memory, we release the gray image cache!
    if (memory_cache != nullptr) {
        memory_cache->img_cache.clear();
    }

    FusionViews views;
    LoadFusionViews(dense_folder, problems, views);
    std::vector<cv::Mat> &images = views.images;
    std::vector<Camera> &cameras = views.cameras;
    std::vector<cv::Mat> &depths = views.depths;
    std::vector<cv::Mat> &normals = views.normals;
    std::vector<cv::Mat> &masks = views.masks;
    std::vector<cv::Mat> &weaks = views.weaks;
    std::vector<cv::Mat> &confidences = views.confidences;
    std::vector<cv::Mat> &skip_weaks = views.skip_weaks;
    const std::unordered_map<int, int> &imageIdToindexMap = views.imageIdToindexMap;
    const std::vector<std::vector<int>> src_slots = SourceSlots(problems, imageIdToindexMap);

    if (weak_filter) {
        WeakVisFilter(problems, cameras, depths, weaks, confidences, dense_folder, skip_weaks);
    }

    std::vector<PointList> PointCloud;
    PointCloud.clear();

    struct CostData {
        float dist;
        float depth;
        float angle;

        CostData() {
            dist = FLT_MAX;
            depth = FLT_MAX;
            angle = FLT_MAX;
        }
    };

    for (int i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        const auto &problem = problems[i];
        int ref_index = imageIdToindexMap.at(problem.ref_image_id);
        const int cols = depths[ref_index].cols;
        const int rows = depths[ref_index].rows;
        int num_ngb = problem.src_image_ids.size();
        std::vector<CostData> diff(num_ngb, CostData());
        int skip_weak = 0;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (skip_weaks[ref_index].at<uchar>(r, c) == 1) {
                    skip_weak++;
                    continue;
                }
                float ref_depth = depths[ref_index].at<float>(r, c);
                if (ref_depth <= 0.0)
                    continue;
                const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
                float3 consistent_Point = PointX;
                float consistent_Color[3] = {(float) images[ref_index].at<cv::Vec3b>(r, c)[0],
                                             (float) images[ref_index].at<cv::Vec3b>(r, c)[1],
                                             (float) images[ref_index].at<cv::Vec3b>(r, c)[2]};
                // values of this pixel only (upstream kept a source's values from an earlier pixel when the source
                // was skipped here, so a stale match could still count)
                std::fill(diff.begin(), diff.end(), CostData());

                for (int j = 0; j < num_ngb; ++j) {
                    int src_index = src_slots[i][j];
                    if (src_index < 0 || depths[src_index].empty())
                        continue;                            // unknown or unusable source: diff[j] stays unset
                    const int src_cols = depths[src_index].cols;
                    const int src_rows = depths[src_index].rows;
                    float2 point;
                    float proj_depth;
                    ProjectCamera(PointX, cameras[src_index], point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (masks[src_index].at<uchar>(src_r, src_c) == 1)
                            continue;
                        float src_depth = depths[src_index].at<float>(src_r, src_c);
                        if (src_depth <= 0.0)
                            continue;
                        const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
                        float2 tmp_pt;
                        ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);
                        diff[j].dist = reproj_error;
                        diff[j].depth = relative_depth_diff;
                        diff[j].angle = angle;
                    }
                }
                for (int k = 2; k <= num_ngb; ++k) {
                    int count = 0;
                    for (int j = 0; j < num_ngb; ++j) {
                        if (diff[j].dist < k * dist_base && diff[j].depth < k * depth_base) {
                            count++;
                        }
                    }
                    if (count >= k) {
                        PointList point3D;
                        point3D.coord = consistent_Point;
                        point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                        PointCloud.emplace_back(point3D);
                        masks[ref_index].at<uchar>(r, c) = 1;
                        break;
                    }
                }
            }
        }
        printf("skip_weak: %d\n", skip_weak);
    }
    path ply_path = dense_folder / path("APD") / path(name);
    ExportPointCloud(ply_path, PointCloud, export_color);
}