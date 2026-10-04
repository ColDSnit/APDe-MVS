#include "APD.h"

__device__  void sort_small(float* d, const int n)
{
	int j;
	for (int i = 1; i < n; i++) {
		float tmp = d[i];
		for (j = i; j >= 1 && tmp < d[j - 1]; j--)
			d[j] = d[j - 1];
		d[j] = tmp;
	}
}

__device__  void sort_small(uchar* d, const int n)
{
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--)
            d[j] = d[j - 1];
        d[j] = tmp;
    }
}

__device__ void sort_small_weighted(short2* points, float* w, int n)
{
	int j;
	for (int i = 1; i < n; i++) {
		short2 tmp = points[i];
		float tmp_w = w[i];
		for (j = i; j >= 1 && tmp_w < w[j - 1]; j--) {
			points[j] = points[j - 1];
			w[j] = w[j - 1];
		}
		points[j] = tmp;
		w[j] = tmp_w;
	}
}

__device__ void getTopNIndex(const uchar *view_weight, int *top_n_index, int n)
{
    // copy view_weight to temp
    float temp[MAX_IMAGES];
    for (int i = 0; i < MAX_IMAGES; ++i) {
        temp[i] = view_weight[i];
    }
    // sort temp and get top n index
    for (int i = 0; i < n; ++i) {
        int max_index = 0;
        for (int j = 1; j < MAX_IMAGES; ++j) {
            if (temp[j] > temp[max_index]) {
                max_index = j;
            }
        }
        top_n_index[i] = max_index;
        temp[max_index] = 0;
    }
}

__device__ int FindMinCostIndex(const float* costs, const int n)
{
	float min_cost = costs[0];
	int min_cost_idx = 0;
	for (int idx = 1; idx < n; ++idx) {
		if (costs[idx] <= min_cost) {
			min_cost = costs[idx];
			min_cost_idx = idx;
		}
	}
	return min_cost_idx;
}

__device__  void setBit(unsigned int* input, const unsigned int n)
{
	(*input) |= (unsigned int)(1 << n);
}

__device__  void unSetBit(unsigned int* input, const unsigned int n)
{
	(*input) &= (unsigned int)(0xFFFFFFFE << n);
}

__device__  int isSet(unsigned int input, const unsigned int n)
{
	return (input >> n) & 1;
}

__device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4* result)
{
	result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
	result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
	result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
}

__device__ float Vec3DotVec3(const float4 vec1, const float4 vec2)
{
	return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ float Vec3DotVec3(const float3 vec1, const float3 vec2)
{
	return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ float3 Vec3CrossVec3(const float3 vec1, const float3 vec2)
{
	float3 cross_vec;
	cross_vec.x = vec1.y * vec2.z - vec2.y * vec1.z;
	cross_vec.y = -(vec1.x * vec2.z - vec2.x * vec1.z);
	cross_vec.z = vec1.x * vec2.y - vec2.x * vec1.y;
	return cross_vec;
}

__device__ float Vec2DotVec2(float2 a, float2 b) {
	return a.x * b.x + a.y * b.y;
}

__device__ float Vec2CrossVec2(float2 a, float2 b) {
	return a.x * b.y - a.y * b.x;
}

__device__ bool PointinTriangle(short2 A, short2 B, short2 C, int2 P)
{
	float2 AB = make_float2(B.x - A.x, B.y - A.y);
	float2 BC = make_float2(C.x - B.x, C.y - B.y);
	float2 CA = make_float2(A.x - C.x, A.y - C.y);
	float AB_ = sqrt(AB.x * AB.x + AB.y * AB.y);
	float BC_ = sqrt(BC.x * BC.x + BC.y * BC.y);
	float CA_ = sqrt(CA.x * CA.x + CA.y * CA.y);
	if (AB_ <= 2 || BC_ <= 2 || CA_ <= 2) {
		return false;
	}
	if (!(AB_ + BC_ > CA_ && BC_ + CA_ > AB_ && AB_ + CA_ > BC_)) {
		return false;
	}
	float2 PA = make_float2(A.x - P.x, A.y - P.y);
	float2 PB = make_float2(B.x - P.x, B.y - P.y);
	float2 PC = make_float2(C.x - P.x, C.y - P.y);
	float t1 = Vec2CrossVec2(PA, PB);
	float t2 = Vec2CrossVec2(PB, PC);
	float t3 = Vec2CrossVec2(PC, PA);
	return t1 * t2 >= 0 && t1 * t3 >= 0;
}

__device__ float TriangleArea(float3 A, float3 B, float3 C)
{
	float3 AB = make_float3(B.x - A.x, B.y - A.y, B.z - A.z);
	float3 BC = make_float3(C.x - B.x, C.y - B.y, C.z - B.z);
	float3 CA = make_float3(A.x - C.x, A.y - C.y, A.z - C.z);
	float AB_ = sqrt(AB.x * AB.x + AB.y * AB.y + AB.z * AB.z);
	float BC_ = sqrt(BC.x * BC.x + BC.y * BC.y + BC.z * BC.z);
	float CA_ = sqrt(CA.x * CA.x + CA.y * CA.y + CA.z * CA.z);
	float P = (AB_ + BC_ + CA_) / 2.0f;
	return sqrt(P * (P - AB_) * (P - BC_) * (P - CA_));
}

__device__ void NormalizeVec3(float4* vec)
{
	const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
	const float inverse_sqrt = rsqrtf(normSquared);
	vec->x *= inverse_sqrt;
	vec->y *= inverse_sqrt;
	vec->z *= inverse_sqrt;
}

__device__ void NormalizeVec2(float2* vec)
{
	const float normSquared = vec->x * vec->x + vec->y * vec->y;
	const float inverse_sqrt = rsqrtf(normSquared);
	vec->x *= inverse_sqrt;
	vec->y *= inverse_sqrt;
}

__device__ void TransformPDFToCDF(float* probs, const int num_probs)
{
	float prob_sum = 0.0f;
	for (int i = 0; i < num_probs; ++i) {
		prob_sum += probs[i];
	}
	const float inv_prob_sum = 1.0f / prob_sum;

	float cum_prob = 0.0f;
	for (int i = 0; i < num_probs; ++i) {
		const float prob = probs[i] * inv_prob_sum;
		cum_prob += prob;
		probs[i] = cum_prob;
	}
}

// Orthographic (telecentric) cameras: the lateral position of a pixel's ray does not depend on depth, so the
// back-projection scale is the fixed reference depth instead of the depth itself.
__device__ void Get3DPoint(const Camera& camera, const int2 p, const float depth, float* X)
{
	const float scale = (camera.model == CAMERA_ORTHOGRAPHIC ? camera.ref_depth : depth);
	X[0] = scale * (p.x - camera.K[2]) / camera.K[0];
	X[1] = scale * (p.y - camera.K[5]) / camera.K[4];
	X[2] = depth;
}

__device__ void Get3DPoint(const Camera& camera, const short2 p, const float depth, float* X)
{
	const float scale = (camera.model == CAMERA_ORTHOGRAPHIC ? camera.ref_depth : depth);
	X[0] = scale * (p.x - camera.K[2]) / camera.K[0];
	X[1] = scale * (p.y - camera.K[5]) / camera.K[4];
	X[2] = depth;
}

__device__ float4 GetViewDirection(const Camera& camera, const int2 p, const float depth)
{
	if (camera.model == CAMERA_ORTHOGRAPHIC) {
		return make_float4(0.0f, 0.0f, 1.0f, 0.0f);  // all rays are parallel to the optical axis
	}
	float X[3];
	Get3DPoint(camera, p, depth, X);
	float norm = sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

	float4 view_direction;
	view_direction.x = X[0] / norm;
	view_direction.y = X[1] / norm;
	view_direction.z = X[2] / norm;
	view_direction.w = 0;
	return view_direction;
}

__device__ float GetDistance2Origin(const Camera& camera, const int2 p, const float depth, const float4 normal)
{
	float X[3];
	Get3DPoint(camera, p, depth, X);
	return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
}

__device__   float SpatialGauss(float x1, float y1, float x2, float y2, float sigma, float mu = 0.0)
{
	float dis = pow(x1 - x2, 2) + pow(y1 - y2, 2) - mu;
	return exp(-1.0 * dis / (2 * sigma * sigma));
}

__device__  float RangeGauss(float x, float sigma, float mu = 0.0)
{
	float x_p = x - mu;
	return exp(-1.0 * (x_p * x_p) / (2 * sigma * sigma));
}

__device__ float ComputeDepthfromPlaneHypothesis(const Camera& camera, const float4 plane_hypothesis, const int2 p)
{
	if (camera.model == CAMERA_ORTHOGRAPHIC) {
		// plane n.X + d = 0 with X = (a, b, z): a and b are fixed by the pixel, solve for z
		const float a = camera.ref_depth * (p.x - camera.K[2]) / camera.K[0];
		const float b = camera.ref_depth * (p.y - camera.K[5]) / camera.K[4];
		return -(plane_hypothesis.w + plane_hypothesis.x * a + plane_hypothesis.y * b) / plane_hypothesis.z;
	}
	return -plane_hypothesis.w * camera.K[0] / ((p.x - camera.K[2]) * plane_hypothesis.x + (camera.K[0] / camera.K[4]) * (p.y - camera.K[5]) * plane_hypothesis.y + camera.K[0] * plane_hypothesis.z);
}

__device__ float4 GenerateRandomNormal(const Camera& camera, const int2 p, curandState* rand_state, const float depth)
{
	float4 normal;
	float q1 = 1.0f;
	float q2 = 1.0f;
	float s = 2.0f;
	while (s >= 1.0f) {
		q1 = 2.0f * curand_uniform(rand_state) - 1.0f;
		q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
		s = q1 * q1 + q2 * q2;
	}
	const float sq = sqrt(1.0f - s);
	normal.x = 2.0f * q1 * sq;
	normal.y = 2.0f * q2 * sq;
	normal.z = 1.0f - 2.0f * s;
	normal.w = 0;

	float4 view_direction = GetViewDirection(camera, p, depth);
	float dot_product = normal.x * view_direction.x + normal.y * view_direction.y + normal.z * view_direction.z;
	if (dot_product > 0.0f) {
		normal.x = -normal.x;
		normal.y = -normal.y;
		normal.z = -normal.z;
	}
	NormalizeVec3(&normal);
	return normal;
}

__device__ float4 GeneratePerturbedNormal(const Camera& camera, const int2 p, const float4 normal, curandState* rand_state, const float perturbation)
{
	float4 view_direction = GetViewDirection(camera, p, 1.0f);

	const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
	const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
	const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

	const float sin_a1 = sin(a1);
	const float sin_a2 = sin(a2);
	const float sin_a3 = sin(a3);
	const float cos_a1 = cos(a1);
	const float cos_a2 = cos(a2);
	const float cos_a3 = cos(a3);

	float R[9];
	R[0] = cos_a2 * cos_a3;
	R[1] = cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3;
	R[2] = sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2;
	R[3] = cos_a2 * sin_a3;
	R[4] = cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3;
	R[5] = cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1;
	R[6] = -sin_a2;
	R[7] = cos_a2 * sin_a1;
	R[8] = cos_a1 * cos_a2;

	float4 normal_perturbed;
	Mat33DotVec3(R, normal, &normal_perturbed);

	if (Vec3DotVec3(normal_perturbed, view_direction) >= 0.0f) {
		normal_perturbed = normal;
	}

	NormalizeVec3(&normal_perturbed);
	return normal_perturbed;
}

// Search bounds of one pixel: its prior band when a valid one is loaded (prior.enable), else the view's range.
__device__ __forceinline__ void PixelDepthBounds(const DataPassHelper* helper, const int center, float* lo, float* hi)
{
	*lo = helper->params->depth_min;
	*hi = helper->params->depth_max;
	if (helper->prior_bounds_cuda != nullptr) {
		const float2 b = helper->prior_bounds_cuda[center];
		if (b.y > b.x) {                     // hi <= lo marks "no prior here"
			*lo = b.x;
			*hi = b.y;
		}
	}
}

// True when depth lies inside the pixel's search bounds (identical to the view-range test without a prior).
__device__ __forceinline__ bool DepthInPixelBounds(const DataPassHelper* helper, const int center, const float depth)
{
	float lo, hi;
	PixelDepthBounds(helper, center, &lo, &hi);
	return depth >= lo && depth <= hi;
}

// Perturbed depth for hypothesis refinement, one uniform draw. With the default options this is the upstream
// expression (+-depth_perturbation * depth, whose && retry loop never repeats). Otherwise the window is local
// (relative, a fraction of the pixel's search range, or absolute) and clipped to [lo, hi].
__device__ float PerturbDepth(const PatchMatchParams* params, curandState* rand_state, const float depth_now,
	const float lo, const float hi, const bool prior_on)
{
	const float u = curand_uniform(rand_state);
	const int mode = params->depth_perturbation_mode;
	if (mode == 0 && !params->perturbation_clip && !prior_on) {
		const float depth_min_perturbed = (1 - params->depth_perturbation) * depth_now;   // upstream, unchanged
		const float depth_max_perturbed = (1 + params->depth_perturbation) * depth_now;
		return u * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
	}
	float a, b;
	if (mode == 1) {                         // fraction of the pixel's search range width
		const float half = params->depth_perturbation * (hi - lo);
		a = depth_now - half;
		b = depth_now + half;
	}
	else if (mode == 2) {                    // absolute half-width in world units
		a = depth_now - params->depth_perturbation_abs;
		b = depth_now + params->depth_perturbation_abs;
	}
	else {                                   // relative, as upstream, but clipped
		a = (1 - params->depth_perturbation) * depth_now;
		b = (1 + params->depth_perturbation) * depth_now;
	}
	a = fmaxf(a, lo);
	b = fminf(b, hi);
	if (!(b > a)) {                          // current depth outside the bounds: draw anywhere inside them
		a = lo;
		b = hi;
	}
	return u * (b - a) + a;
}

__device__ float4 GenerateRandomPlaneHypothesis(const Camera& camera, const int2 p, curandState* rand_state, const float depth_min, const float depth_max)
{
	float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
	float4 plane_hypothesis = GenerateRandomNormal(camera, p, rand_state, depth);
	plane_hypothesis.w = GetDistance2Origin(camera, p, depth, plane_hypothesis);
	return plane_hypothesis;
}

__device__ float4 GeneratePertubedPlaneHypothesis(const Camera& camera, const int2 p, curandState* rand_state, const float perturbation, const float4 plane_hypothesis_now, const float depth_now, const float depth_min, const float depth_max)
{
	float depth_perturbed = depth_now;

	float dist_perturbed = plane_hypothesis_now.w;
	const float dist_min_perturbed = (1 - perturbation) * dist_perturbed;
	const float dist_max_perturbed = (1 + perturbation) * dist_perturbed;
	float4 plane_hypothesis_temp = plane_hypothesis_now;
	do {
		dist_perturbed = curand_uniform(rand_state) * (dist_max_perturbed - dist_min_perturbed) + dist_min_perturbed;
		plane_hypothesis_temp.w = dist_perturbed;
		depth_perturbed = ComputeDepthfromPlaneHypothesis(camera, plane_hypothesis_temp, p);
	} while (depth_perturbed < depth_min && depth_perturbed > depth_max);

	float4 plane_hypothesis = GeneratePerturbedNormal(camera, p, plane_hypothesis_now, rand_state, perturbation * M_PI);
	plane_hypothesis.w = dist_perturbed;
	return plane_hypothesis;
}

__device__ void ComputeHomography(const Camera& ref_camera, const Camera& src_camera, const float4 plane_hypothesis, float* H)
{
	float ref_C[3];
	float src_C[3];
	ref_C[0] = -(ref_camera.R[0] * ref_camera.t[0] + ref_camera.R[3] * ref_camera.t[1] + ref_camera.R[6] * ref_camera.t[2]);
	ref_C[1] = -(ref_camera.R[1] * ref_camera.t[0] + ref_camera.R[4] * ref_camera.t[1] + ref_camera.R[7] * ref_camera.t[2]);
	ref_C[2] = -(ref_camera.R[2] * ref_camera.t[0] + ref_camera.R[5] * ref_camera.t[1] + ref_camera.R[8] * ref_camera.t[2]);
	src_C[0] = -(src_camera.R[0] * src_camera.t[0] + src_camera.R[3] * src_camera.t[1] + src_camera.R[6] * src_camera.t[2]);
	src_C[1] = -(src_camera.R[1] * src_camera.t[0] + src_camera.R[4] * src_camera.t[1] + src_camera.R[7] * src_camera.t[2]);
	src_C[2] = -(src_camera.R[2] * src_camera.t[0] + src_camera.R[5] * src_camera.t[1] + src_camera.R[8] * src_camera.t[2]);

	float R_relative[9];
	float C_relative[3];
	float t_relative[3];
	R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] + src_camera.R[2] * ref_camera.R[2];
	R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] + src_camera.R[2] * ref_camera.R[5];
	R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] + src_camera.R[2] * ref_camera.R[8];
	R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] + src_camera.R[5] * ref_camera.R[2];
	R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] + src_camera.R[5] * ref_camera.R[5];
	R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] + src_camera.R[5] * ref_camera.R[8];
	R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] + src_camera.R[8] * ref_camera.R[2];
	R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] + src_camera.R[8] * ref_camera.R[5];
	R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] + src_camera.R[8] * ref_camera.R[8];
	C_relative[0] = (ref_C[0] - src_C[0]);
	C_relative[1] = (ref_C[1] - src_C[1]);
	C_relative[2] = (ref_C[2] - src_C[2]);
	t_relative[0] = src_camera.R[0] * C_relative[0] + src_camera.R[1] * C_relative[1] + src_camera.R[2] * C_relative[2];
	t_relative[1] = src_camera.R[3] * C_relative[0] + src_camera.R[4] * C_relative[1] + src_camera.R[5] * C_relative[2];
	t_relative[2] = src_camera.R[6] * C_relative[0] + src_camera.R[7] * C_relative[1] + src_camera.R[8] * C_relative[2];

	if (ref_camera.model == CAMERA_ORTHOGRAPHIC) {
		// Between two orthographic cameras a plane induces an affine map. A reference pixel (px, py) has
		// camera coordinates a = ax*px + bx, b = ay*py + by and, on the plane, z = z0 + zx*px + zy*py.
		const float ax = ref_camera.ref_depth / ref_camera.K[0];
		const float ay = ref_camera.ref_depth / ref_camera.K[4];
		const float bx = -ax * ref_camera.K[2];
		const float by = -ay * ref_camera.K[5];
		const float inv_nz = 1.0f / plane_hypothesis.z;
		const float z0 = -(plane_hypothesis.w + plane_hypothesis.x * bx + plane_hypothesis.y * by) * inv_nz;
		const float zx = -plane_hypothesis.x * ax * inv_nz;
		const float zy = -plane_hypothesis.y * ay * inv_nz;
		// source magnification in pixels per world unit
		const float mx = src_camera.K[0] / src_camera.ref_depth;
		const float my = src_camera.K[4] / src_camera.ref_depth;
		H[0] = mx * (R_relative[0] * ax + R_relative[2] * zx);
		H[1] = mx * (R_relative[1] * ay + R_relative[2] * zy);
		H[2] = mx * (R_relative[0] * bx + R_relative[1] * by + R_relative[2] * z0 + t_relative[0]) + src_camera.K[2];
		H[3] = my * (R_relative[3] * ax + R_relative[5] * zx);
		H[4] = my * (R_relative[4] * ay + R_relative[5] * zy);
		H[5] = my * (R_relative[3] * bx + R_relative[4] * by + R_relative[5] * z0 + t_relative[1]) + src_camera.K[5];
		H[6] = 0.0f;
		H[7] = 0.0f;
		H[8] = 1.0f;
		return;
	}

	H[0] = R_relative[0] - t_relative[0] * plane_hypothesis.x / plane_hypothesis.w;
	H[1] = R_relative[1] - t_relative[0] * plane_hypothesis.y / plane_hypothesis.w;
	H[2] = R_relative[2] - t_relative[0] * plane_hypothesis.z / plane_hypothesis.w;
	H[3] = R_relative[3] - t_relative[1] * plane_hypothesis.x / plane_hypothesis.w;
	H[4] = R_relative[4] - t_relative[1] * plane_hypothesis.y / plane_hypothesis.w;
	H[5] = R_relative[5] - t_relative[1] * plane_hypothesis.z / plane_hypothesis.w;
	H[6] = R_relative[6] - t_relative[2] * plane_hypothesis.x / plane_hypothesis.w;
	H[7] = R_relative[7] - t_relative[2] * plane_hypothesis.y / plane_hypothesis.w;
	H[8] = R_relative[8] - t_relative[2] * plane_hypothesis.z / plane_hypothesis.w;

	float tmp[9];
	tmp[0] = H[0] / ref_camera.K[0];
	tmp[1] = H[1] / ref_camera.K[4];
	tmp[2] = -H[0] * ref_camera.K[2] / ref_camera.K[0] - H[1] * ref_camera.K[5] / ref_camera.K[4] + H[2];
	tmp[3] = H[3] / ref_camera.K[0];
	tmp[4] = H[4] / ref_camera.K[4];
	tmp[5] = -H[3] * ref_camera.K[2] / ref_camera.K[0] - H[4] * ref_camera.K[5] / ref_camera.K[4] + H[5];
	tmp[6] = H[6] / ref_camera.K[0];
	tmp[7] = H[7] / ref_camera.K[4];
	tmp[8] = -H[6] * ref_camera.K[2] / ref_camera.K[0] - H[7] * ref_camera.K[5] / ref_camera.K[4] + H[8];

	H[0] = src_camera.K[0] * tmp[0] + src_camera.K[2] * tmp[6];
	H[1] = src_camera.K[0] * tmp[1] + src_camera.K[2] * tmp[7];
	H[2] = src_camera.K[0] * tmp[2] + src_camera.K[2] * tmp[8];
	H[3] = src_camera.K[4] * tmp[3] + src_camera.K[5] * tmp[6];
	H[4] = src_camera.K[4] * tmp[4] + src_camera.K[5] * tmp[7];
	H[5] = src_camera.K[4] * tmp[5] + src_camera.K[5] * tmp[8];
	H[6] = src_camera.K[8] * tmp[6];
	H[7] = src_camera.K[8] * tmp[7];
	H[8] = src_camera.K[8] * tmp[8];
}

__device__ float2 ComputeCorrespondingPoint(const float* H, const int2 p)
{
	float3 pt;
	pt.x = H[0] * p.x + H[1] * p.y + H[2];
	pt.y = H[3] * p.x + H[4] * p.y + H[5];
	pt.z = H[6] * p.x + H[7] * p.y + H[8];
	return make_float2(pt.x / pt.z, pt.y / pt.z);
}

__device__ float4 TransformNormal(const Camera& camera, float4 plane_hypothesis)
{
	float4 transformed_normal;
	transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[3] * plane_hypothesis.y + camera.R[6] * plane_hypothesis.z;
	transformed_normal.y = camera.R[1] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[7] * plane_hypothesis.z;
	transformed_normal.z = camera.R[2] * plane_hypothesis.x + camera.R[5] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
	transformed_normal.w = plane_hypothesis.w;
	return transformed_normal;
}

__device__ float4 TransformNormal2RefCam(const Camera& camera, float4 plane_hypothesis)
{
	float4 transformed_normal;
	transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[1] * plane_hypothesis.y + camera.R[2] * plane_hypothesis.z;
	transformed_normal.y = camera.R[3] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[5] * plane_hypothesis.z;
	transformed_normal.z = camera.R[6] * plane_hypothesis.x + camera.R[7] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
	transformed_normal.w = plane_hypothesis.w;
	return transformed_normal;
}

__device__ short2 GetAnchorPoint(const int2 p, const int index, const DataPassHelper* helper) {
	const unsigned offset = helper->anchors_map_cuda[p.x + p.y * helper->width] * ANCHOR_NUM;
	short2 anchor_pt = helper->anchors_cuda[offset + index];
	return anchor_pt;
}

__device__ void Softmax(float *costs, int n) {
    float max_cost = -1e10;
    for (int i = 0; i < n; i++) {
        if (costs[i] > max_cost) {
            max_cost = costs[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < n; i++) {
        costs[i] = exp(costs[i] - max_cost);
        sum += costs[i];
    }
    for (int i = 0; i < n; i++) {
        costs[i] /= sum;
    }
}

__device__ float ComputeBilateralNCCNew(
        const int2 p,
        const int src_idx,
        const float4 plane_hypothesis,
        const DataPassHelper* helper
) {
    const cudaTextureObject_t ref_image = helper->texture_objects_cuda[0].images[0];
    const Camera& ref_camera = helper->cameras_cuda[0];
    const cudaTextureObject_t src_image = helper->texture_objects_cuda[0].images[src_idx];
    const Camera& src_camera = helper->cameras_cuda[src_idx];
    const PatchMatchParams* params = helper->params;
    const uchar* weak_info = helper->weak_info_cuda;
    const int width = helper->width;
    const int height = helper->height;
    const int center = p.x + p.y * width;
    const uchar *sa_mask = helper->sa_mask_cuda;
    const uchar center_sa_id = sa_mask[center];
    bool use_sa_mask = (sa_mask[center] != 0);


    const float cost_max = 2.0f;

    float H[9];
    ComputeHomography(ref_camera, src_camera, plane_hypothesis, H);
    float2 pt = ComputeCorrespondingPoint(H, p);
    if (pt.x >= src_camera.width || pt.x < 0.0f || pt.y >= src_camera.height || pt.y < 0.0f) {
        return cost_max;
    }

    float cost = 0.0f;
    float strong_costs[9];
    int strong_costs_num = 0;

    if (weak_info[center] == WEAK) {
        // for weak texture area use deformable ncc
        const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);
        // the strong points
        float center_cost = 0.0f;
        float strong_cost = 0.0f;
        float strong_weight = 0;
        for (int k = 0; k < ANCHOR_NUM; ++k) {
            const short2 anchor_pt = GetAnchorPoint(p, k, helper);
            if (anchor_pt.x == -1 || anchor_pt.y == -1) {
                continue;
            }
            if (use_sa_mask) {
                if (sa_mask[anchor_pt.x + anchor_pt.y * width] != center_sa_id) {
                    continue;
                }
            }

            float2 anchor_src_pt = ComputeCorrespondingPoint(H, make_int2(anchor_pt.x, anchor_pt.y));
            if (anchor_src_pt.x < 0 || anchor_src_pt.y < 0 || anchor_src_pt.x >= width || anchor_src_pt.y >= height) {
                if (k != 0) {
                    unsigned int view_info = helper->selected_views_cuda[anchor_pt.x + anchor_pt.y * width];
                    if (isSet(view_info, src_idx - 1)) {
                        strong_costs[strong_costs_num++] = cost_max;
                        strong_weight += 1;
                    }
                    continue;
                }
                else {
                    return cost_max;
                }
            }
            // compute ncc for this point
            float sum_ref = 0.0f;
            float sum_ref_ref = 0.0f;
            float sum_src = 0.0f;
            float sum_src_src = 0.0f;
            float sum_ref_src = 0.0f;
            float bilateral_weight_sum = 0.0f;
            const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);
            int radius = (k == 0 ? params->strong_radius : params->weak_radius);
            int increment = (k == 0 ? params->strong_increment : params->weak_increment);
            for (int i = -radius; i <= radius; i += increment) {
                for (int j = -radius; j <= radius; j += increment) {
                    const int2 ref_pt = make_int2(anchor_pt.x + i, anchor_pt.y + j);
                    if (use_sa_mask) {
                        if (sa_mask[ref_pt.x + ref_pt.y * width] != center_sa_id) {
                            continue;
                        }
                    }
                    const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                    float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
                    const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
                    float weight = 1.0f;
                    sum_ref += weight * ref_pix;
                    sum_ref_ref += weight * ref_pix * ref_pix;
                    sum_src += weight * src_pix;
                    sum_src_src += weight * src_pix * src_pix;
                    sum_ref_src += weight * ref_pix * src_pix;
                    bilateral_weight_sum += weight;
                }
            }
            if (bilateral_weight_sum == 0.0f) {
                continue;
            }
            const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
            sum_ref *= inv_bilateral_weight_sum;
            sum_ref_ref *= inv_bilateral_weight_sum;
            sum_src *= inv_bilateral_weight_sum;
            sum_src_src *= inv_bilateral_weight_sum;
            sum_ref_src *= inv_bilateral_weight_sum;
            const float var_ref = sum_ref_ref - sum_ref * sum_ref;
            const float var_src = sum_src_src - sum_src * sum_src;
            const float kMinVar = 1e-5f;
            float temp_cost = 0.0f;
            if (var_ref < kMinVar || var_src < kMinVar) {
                temp_cost = cost_max;
            }
            else {
                const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
                const float var_ref_src = sqrt(var_ref * var_src);
                temp_cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
            }
            if (k == 0) {
                center_cost = temp_cost;
            }
            else {
                strong_costs[strong_costs_num++] = temp_cost;
                strong_weight += 1;
            }
        }
        if (strong_weight <= 1e-6f) {
            cost = center_cost;
        }
        else {
            float strong_costs_weight[9];
            for (int i = 0; i < strong_costs_num; ++i) {
                strong_costs_weight[i] = strong_costs[i];
            }
            Softmax(strong_costs_weight, strong_costs_num);
            strong_cost = 0.0f;
            for (int i = 0; i < strong_costs_num; ++i) {
                strong_cost += strong_costs_weight[i] * strong_costs[i];
            }
            strong_cost = MIN(strong_cost, cost_max);
            cost = params->weak_center_weight * center_cost + (1.0 - params->weak_center_weight) * strong_cost;
        }
    }
    else {
        printf("error\n");
    }
    return cost;
}


__device__ float ComputeBilateralNCCOld(
	const int2 p,
	const int src_idx,
	const float4 plane_hypothesis,
	const DataPassHelper* helper
) {
	const cudaTextureObject_t ref_image = helper->texture_objects_cuda[0].images[0];
	const Camera& ref_camera = helper->cameras_cuda[0];
	const cudaTextureObject_t src_image = helper->texture_objects_cuda[0].images[src_idx];
	const Camera& src_camera = helper->cameras_cuda[src_idx];
    const uchar *sa_mask = helper->sa_mask_cuda;
    const uchar center_sa_id = sa_mask[p.x + p.y * ref_camera.width];

	const float cost_max = 2.0f;

	float H[9];
	ComputeHomography(ref_camera, src_camera, plane_hypothesis, H);
	float2 pt = ComputeCorrespondingPoint(H, p);
	if (pt.x >= src_camera.width || pt.x < 0.0f || pt.y >= src_camera.height || pt.y < 0.0f) {
		return cost_max;
	}
    const int radius = helper->params->strong_radius;
	const int increment = helper->params->strong_increment;
    const int center = pt.y * src_camera.width + pt.x;
	float cost = 0.0f;
    if (sa_mask[center] == 0) {
		float sum_ref = 0.0f;
		float sum_ref_ref = 0.0f;
		float sum_src = 0.0f;
		float sum_src_src = 0.0f;
		float sum_ref_src = 0.0f;
		float bilateral_weight_sum = 0.0f;

		for (int i = -radius; i <= radius; i += increment) {
			for (int j = -radius; j <= radius; j += increment) {
				const int2 ref_pt = make_int2(p.x + i, p.y + j);
				const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
				float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
				const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
				float weight = 1.0f;
				sum_ref += weight * ref_pix;
				sum_ref_ref += weight * ref_pix * ref_pix;
				sum_src += weight * src_pix;
				sum_src_src += weight * src_pix * src_pix;
				sum_ref_src += weight * ref_pix * src_pix;
				bilateral_weight_sum += weight;
			}
        }
		const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
		sum_ref *= inv_bilateral_weight_sum;
		sum_ref_ref *= inv_bilateral_weight_sum;
		sum_src *= inv_bilateral_weight_sum;
		sum_src_src *= inv_bilateral_weight_sum;
		sum_ref_src *= inv_bilateral_weight_sum;

		const float var_ref = sum_ref_ref - sum_ref * sum_ref;
		const float var_src = sum_src_src - sum_src * sum_src;

		const float kMinVar = 1e-5f;
		if (var_ref < kMinVar || var_src < kMinVar) {
			cost = cost_max;
		}
		else {
			const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
			const float var_ref_src = sqrt(var_ref * var_src);
			cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
		}
	}
    else {
        const int sign[] = {1, 1, -1, -1, 1, -1, -1, 1};
        const int offset[] = {1, 1, 3, 1, 1, 3, 1, 5, 3, 3, 5, 1, 5, 3, 3, 5, 5, 5};
        float sum_ref = 0.0f;
        float sum_ref_ref = 0.0f;
        float sum_src = 0.0f;
        float sum_src_src = 0.0f;
        float sum_ref_src = 0.0f;
        float bilateral_weight_sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int x_sign = sign[i * 2];
            const int y_sign = sign[i * 2 + 1];
            for (int j = 0; j < 9; ++j) {
                const int x_offset = offset[j * 2] * x_sign;
                const int y_offset = offset[j * 2 + 1] * y_sign;
                const int2 ref_pt = make_int2(p.x + x_offset, p.y + y_offset);
                if (ref_pt.x < 0 || ref_pt.x >= ref_camera.width || ref_pt.y < 0 || ref_pt.y >= ref_camera.height) {
                    continue;
                }
                const int ref_idx = ref_pt.y * ref_camera.width + ref_pt.x;
                if (sa_mask[ref_idx] != center_sa_id) {
                    break;
                }
                const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
                const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
                float weight = 1.0f;
                sum_ref += weight * ref_pix;
                sum_ref_ref += weight * ref_pix * ref_pix;
                sum_src += weight * src_pix;
                sum_src_src += weight * src_pix * src_pix;
                sum_ref_src += weight * ref_pix * src_pix;
                bilateral_weight_sum += weight;
            }
        }
        const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
        sum_ref *= inv_bilateral_weight_sum;
        sum_ref_ref *= inv_bilateral_weight_sum;
        sum_src *= inv_bilateral_weight_sum;
        sum_src_src *= inv_bilateral_weight_sum;
        sum_ref_src *= inv_bilateral_weight_sum;

        const float var_ref = sum_ref_ref - sum_ref * sum_ref;
        const float var_src = sum_src_src - sum_src * sum_src;

        const float kMinVar = 1e-5f;
        if (var_ref < kMinVar || var_src < kMinVar) {
            cost = cost_max;
        }
        else {
            const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
            const float var_ref_src = sqrt(var_ref * var_src);
            cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
        }

    }
	return cost;
}

__device__ float ComputeMultiViewInitialCostandSelectedViews(
	const int2 p,
	DataPassHelper* helper
) {
	PatchMatchParams* params = helper->params;
	unsigned int* selected_views = helper->selected_views_cuda;
	int center = p.x + p.y * helper->width;
	float4 plane_hypothesis = helper->plane_hypotheses_cuda[center];
	uchar *weak_info = helper->weak_info_cuda;
	
	float cost_max = 2.0f;
	float cost_vector[32] = { 2.0f };
	float cost_vector_copy[32] = { 2.0f };
	int cost_count = 0;
	int num_valid_views = 0;

	for (int i = 1; i < params->num_images; ++i) {
		float c = 0.0f;
		if (params->use_APD && weak_info[center] == WEAK) {
			c = ComputeBilateralNCCNew(p, i, plane_hypothesis, helper);
		} else {
			c = ComputeBilateralNCCOld(p, i, plane_hypothesis, helper);
		}
		cost_vector[i - 1] = c;
		cost_vector_copy[i - 1] = c;
		cost_count++;
		if (c < cost_max) {
			num_valid_views++;
		}
	}

	sort_small(cost_vector, cost_count);
	selected_views[center] = 0;

	int top_k = min(num_valid_views, params->top_k);
	if (top_k > 0) {
		float cost = 0.0f;
		for (int i = 0; i < top_k; ++i) {
			cost += cost_vector[i];
		}
		float cost_threshold = cost_vector[top_k - 1];
		for (int i = 0; i < params->num_images - 1; ++i) {
			if (cost_vector_copy[i] <= cost_threshold) {
				setBit(&(selected_views[center]), i);
			}
		}
		return cost / top_k;
	}
	else {
		return cost_max;
	}
}

__device__ float ComputeMultiViewInitialCost(
	const int2 p,
	DataPassHelper* helper
) {
	PatchMatchParams* params = helper->params;
	unsigned int* selected_views = helper->selected_views_cuda;
	int center = p.x + p.y * helper->width;
	float4 plane_hypothesis = helper->plane_hypotheses_cuda[center];

	const float cost_max = 2.0f;
	int cost_count = 0;
	float cost = 0.0f;

	for (int i = 1; i < params->num_images; ++i) {
		if (isSet(selected_views[center], i - 1)) {
			float c = ComputeBilateralNCCOld(p, i, plane_hypothesis, helper);
			if (c < cost_max) {
				cost_count++;
				cost += c;
			}
			else {
				unSetBit(&(selected_views[center]), i - 1);
			}
		}
	}
	if (cost_count == 0) {
		return cost_max;
	}
	else {
		return cost / cost_count;
	}
}

__device__ void ComputeMultiViewCostVectorNew(
	const int2 p,
	float4 plane_hypothesis,
	float* cost_vector,
	DataPassHelper* helper
) {
	for (int i = 1; i < helper->params->num_images; ++i) {
		cost_vector[i - 1] = ComputeBilateralNCCNew(p, i, plane_hypothesis, helper);
	}
}

__device__ void ComputeMultiViewCostVectorOld(
	const int2 p,
	float4 plane_hypothesis,
	float* cost_vector,
	DataPassHelper* helper
) {
	for (int i = 1; i < helper->params->num_images; ++i) {
		cost_vector[i - 1] = ComputeBilateralNCCOld(p, i, plane_hypothesis, helper);
	}
}

__device__ float3 Get3DPointonWorld_cu(const float x, const float y, const float depth, const Camera& camera)
{
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
	pointX.x = tmpX.x + camera.c[0];
	pointX.y = tmpX.y + camera.c[1];
	pointX.z = tmpX.z + camera.c[2];

	return pointX;
}

__device__ void ProjectonCamera_cu(const float3 PointX, const Camera& camera, float2& point, float& depth)
{
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

__device__ float ComputeGeomConsistencyCost(
	const int2 p,
	const int src_idx,
	const float4 plane_hypothesis,
	DataPassHelper* helper
) {
	const Camera& ref_camera = helper->cameras_cuda[0];
	const Camera& src_camera = helper->cameras_cuda[src_idx];
	const cudaTextureObject_t depth_image = helper->texture_depths_cuda[0].images[src_idx];

	const float max_cost = helper->params->geom_max_cost;

	float center_cost = 0.0f;
	{
		float depth = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, p);
		float3 forward_point = Get3DPointonWorld_cu(p.x, p.y, depth, ref_camera);

		float2 src_pt;
		float src_d;
		ProjectonCamera_cu(forward_point, src_camera, src_pt, src_d);
		const float src_depth = tex2D<float>(depth_image, (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);

		if (src_depth == 0.0f) {
			return max_cost;
		}

		float3 src_3D_pt = Get3DPointonWorld_cu(src_pt.x, src_pt.y, src_depth, src_camera);

		float2 backward_point;
		float ref_d;
		ProjectonCamera_cu(src_3D_pt, ref_camera, backward_point, ref_d);

		const float diff_col = p.x - backward_point.x;
		const float diff_row = p.y - backward_point.y;
		center_cost = sqrt(diff_col * diff_col + diff_row * diff_row);
	}
	return min(max_cost, center_cost);
}

__global__ void InitRandomStates(
	DataPassHelper* helper
) {
	const int width = helper->width;
	const int height = helper->height;
	curandState* rand_states = helper->rand_states_cuda;

	const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	// rng_seed < 0 keeps the upstream clock seed (runs differ); a fixed seed makes a run repeatable
	const long long rng_seed = helper->params->rng_seed;
	curand_init(rng_seed < 0 ? clock64() : rng_seed, p.y, p.x, &rand_states[center]);
}

__global__ void RandomInitialization(
	DataPassHelper* helper
) {
	int width = helper->width;
	int height = helper->height;
	const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	Camera* cameras = helper->cameras_cuda;
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	float* costs = helper->costs_cuda;
	curandState* rand_states = helper->rand_states_cuda;
	PatchMatchParams* params = helper->params;

	float depth_lo, depth_hi;
	PixelDepthBounds(helper, center, &depth_lo, &depth_hi);       // view range, or this pixel's prior band
	if (params->state == FIRST_INIT) {
		plane_hypotheses[center] = GenerateRandomPlaneHypothesis(cameras[0], p, &rand_states[center], depth_lo, depth_hi);
		costs[center] = ComputeMultiViewInitialCostandSelectedViews(p, helper);
	}
	else {
		float4 plane_hypothesis;
		plane_hypothesis = plane_hypotheses[center];
		plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);
		float depth = plane_hypothesis.w;
		if (helper->prior_bounds_cuda != nullptr && (depth < depth_lo || depth > depth_hi)) {
			// prior on: an inherited depth outside the band (or 0 = none) is re-seeded randomly inside it
			plane_hypothesis = GenerateRandomPlaneHypothesis(cameras[0], p, &rand_states[center], depth_lo, depth_hi);
		}
		else {
			plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
		}
		plane_hypotheses[center] = plane_hypothesis;
		costs[center] = ComputeMultiViewInitialCostandSelectedViews(p, helper);
	}
}

__device__ void PlaneHypothesisRefinementStrong(
	float4* plane_hypothesis,
	float* depth,
	float* cost,
	curandState* rand_state,
	const uchar* view_weights,
	const float weight_norm,
	const int2 p,
	DataPassHelper* helper

) {
	float depth_perturbation = helper->params->depth_perturbation;
	float normal_perturbation = helper->params->normal_perturbation;
	const Camera* cameras = helper->cameras_cuda;
	const PatchMatchParams* params = helper->params;
	float depth_min = params->depth_min;
	float depth_max = params->depth_max;
	PixelDepthBounds(helper, p.y * helper->width + p.x, &depth_min, &depth_max);   // prior band if loaded

	float depth_rand = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
	float4 plane_hypothesis_rand = GenerateRandomNormal(cameras[0], p, rand_state, *depth);
	(void)depth_perturbation;
	const float depth_perturbed = PerturbDepth(params, rand_state, *depth, depth_min, depth_max, helper->prior_bounds_cuda != nullptr);
	float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, normal_perturbation * M_PI);

	const int num_planes = 5;
	float depths[num_planes] = { depth_rand, *depth, depth_rand, *depth, depth_perturbed };
	float4 normals[num_planes] = { *plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis };

	for (int i = 0; i < num_planes; ++i) {
		float cost_vector[32] = { 2.0f };
		float4 temp_plane_hypothesis = normals[i];
		temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depths[i], temp_plane_hypothesis);
		ComputeMultiViewCostVectorOld(p, temp_plane_hypothesis, cost_vector, helper);

		float temp_cost = 0.0f;
		for (int j = 0; j < params->num_images - 1; ++j) {
			if (params->geom_consistency && params->use_impetus) {
				temp_cost += view_weights[j] * (cost_vector[j] + params->geom_factor * ComputeGeomConsistencyCost(p, j + 1, temp_plane_hypothesis, helper));
			}
			else {
				temp_cost += view_weights[j] * cost_vector[j];
			}
		}
		temp_cost /= weight_norm;

		float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], temp_plane_hypothesis, p);
		if (depth_before >= depth_min && depth_before <= depth_max && temp_cost < *cost) {
			*depth = depth_before;
			*plane_hypothesis = temp_plane_hypothesis;
			*cost = temp_cost;
		}
	}
}

__device__ void PlaneHypothesisRefinementWeak(
	float4* plane_hypothesis,
	float* depth,
	float* cost,
	curandState* rand_state,
	const uchar* view_weights,
	const float weight_norm,
	const int2 p,
	DataPassHelper* helper

) {
	float depth_perturbation = helper->params->depth_perturbation;
	float normal_perturbation = helper->params->normal_perturbation;
	const Camera* cameras = helper->cameras_cuda;
	const PatchMatchParams* params = helper->params;
	float depth_min = params->depth_min;
	float depth_max = params->depth_max;
	const int center = p.x + p.y * helper->width;
	PixelDepthBounds(helper, center, &depth_min, &depth_max);    // prior band if loaded
	{   // test the fit plane
		float4 fit_plane_hypothesis = helper->fit_plane_hypotheses_cuda[center];
		if (fit_plane_hypothesis.x == 0 && fit_plane_hypothesis.y == 0 && fit_plane_hypothesis.z == 0) {
			return;
		}
		float cost_vector[32] = { 2.0f };
		ComputeMultiViewCostVectorNew(p, fit_plane_hypothesis, cost_vector, helper);
		float temp_cost = 0.0f;
		for (int j = 0; j < params->num_images - 1; ++j) {
			if (view_weights[j] > 0) {
				if (params->geom_consistency) {
					temp_cost += view_weights[j] * (cost_vector[j] + params->geom_factor * ComputeGeomConsistencyCost(p, j + 1, fit_plane_hypothesis, helper));
				}
				else {
					temp_cost += view_weights[j] * cost_vector[j];
				}
			}
		}
		temp_cost /= weight_norm;

		float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], fit_plane_hypothesis, p);
		if (depth_before >= depth_min && depth_before <= depth_max && temp_cost < *cost) {
			*depth = depth_before;
			*plane_hypothesis = fit_plane_hypothesis;
			*cost = temp_cost;
		}
	}
	// random refine
	{
		float depth_rand = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
		float4 plane_hypothesis_rand = GenerateRandomNormal(cameras[0], p, rand_state, *depth);
		(void)depth_perturbation;
		const float depth_perturbed = PerturbDepth(params, rand_state, *depth, depth_min, depth_max, helper->prior_bounds_cuda != nullptr);
		float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, normal_perturbation * M_PI);

		const int num_planes = 5;
		float depths[num_planes] = { depth_rand, *depth, depth_rand, *depth, depth_perturbed };
		float4 normals[num_planes] = { *plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis };

		for (int i = 0; i < num_planes; ++i) {
			float cost_vector[32] = { 2.0f };
			float4 temp_plane_hypothesis = normals[i];
			temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depths[i], temp_plane_hypothesis);
			ComputeMultiViewCostVectorNew(p, temp_plane_hypothesis, cost_vector, helper);

			float temp_cost = 0.0f;
			for (int j = 0; j < params->num_images - 1; ++j) {
				if (view_weights[j] > 0) {
					if (params->geom_consistency) {
						temp_cost += view_weights[j] * (cost_vector[j] + params->geom_factor * ComputeGeomConsistencyCost(p, j + 1, temp_plane_hypothesis, helper));
					}
					else {
						temp_cost += view_weights[j] * cost_vector[j];
					}
				}
			}
			temp_cost /= weight_norm;

			float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], temp_plane_hypothesis, p);
			if (depth_before >= depth_min && depth_before <= depth_max && temp_cost < *cost) {
				*depth = depth_before;
				*plane_hypothesis = temp_plane_hypothesis;
				*cost = temp_cost;
			}
		}
	}
}

__device__ void CheckerboardPropagationStrong(
	const int2 p,
	const int iter,
	DataPassHelper* helper
) {
	const int width = helper->width;
	const int height = helper->height;
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	float* costs = helper->costs_cuda;
	curandState* rand_states = helper->rand_states_cuda;
	unsigned int* selected_views = helper->selected_views_cuda;
	PatchMatchParams* params = helper->params;
	const Camera* cameras = helper->cameras_cuda;
	int num_images = params->num_images;

	if (p.x >= width || p.y >= height) {
		return;
	}

	const int center = p.y * width + p.x;

	// Adaptive Checkerboard Sampling
	float cost_array[8][32] = { 2.0f };
	bool flag[8] = { false };
	int num_valid_pixels = 0;

	float costMin;
	int costMinPoint;

	int left_near = center - 1;
	int left_far = center - 3;
	int right_near = center + 1;
	int right_far = center + 3;
	int up_near = center - width;
	int up_far = center - 3 * width;
	int down_near = center + width;
	int down_far = center + 3 * width;
	// 0 -- up_near, 1 -- up_far, 2 -- down_near, 3 -- down_far, 4 -- left_near, 5 -- left_far, 6 -- right_near, 7 -- right_far
	// up_far
	if (p.y > 2) {
		flag[1] = true;
		num_valid_pixels++;
		costMin = costs[up_far];
		costMinPoint = up_far;
		for (int i = 1; i < 11; ++i) {
			if (p.y > 2 + 2 * i) {
				int pointTemp = up_far - 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		up_far = costMinPoint;
		ComputeMultiViewCostVectorOld(p, plane_hypotheses[up_far], cost_array[1], helper);
	}

	// dwon_far
	if (p.y < height - 3) {
		flag[3] = true;
		num_valid_pixels++;
		costMin = costs[down_far];
		costMinPoint = down_far;
		for (int i = 1; i < 11; ++i) {
			if (p.y < height - 3 - 2 * i) {
				int pointTemp = down_far + 2 * i * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		down_far = costMinPoint;
		ComputeMultiViewCostVectorOld(p, plane_hypotheses[down_far], cost_array[3], helper);
	}

	// left_far
	if (p.x > 2) {
		flag[5] = true;
		num_valid_pixels++;
		costMin = costs[left_far];
		costMinPoint = left_far;
		for (int i = 1; i < 11; ++i) {
			if (p.x > 2 + 2 * i) {
				int pointTemp = left_far - 2 * i;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		left_far = costMinPoint;
		ComputeMultiViewCostVectorOld(p, plane_hypotheses[left_far], cost_array[5], helper);
	}

	// right_far
	if (p.x < width - 3) {
		flag[7] = true;
		num_valid_pixels++;
		costMin = costs[right_far];
		costMinPoint = right_far;
		for (int i = 1; i < 11; ++i) {
			if (p.x < width - 3 - 2 * i) {
				int pointTemp = right_far + 2 * i;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		right_far = costMinPoint;
		ComputeMultiViewCostVectorOld(p, plane_hypotheses[right_far], cost_array[7], helper);
	}

	// up_near
	if (p.y > 0) {
		flag[0] = true;
		num_valid_pixels++;
		costMin = costs[up_near];
		costMinPoint = up_near;
		for (int i = 0; i < 3; ++i) {
			if (p.y > 1 + i && p.x > i) {
				int pointTemp = up_near - (1 + i) * width - (i + 1);
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.y > 1 + i && p.x < width - 1 - i) {
				int pointTemp = up_near - (1 + i) * width + (i + 1);
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		up_near = costMinPoint;
		ComputeMultiViewCostVectorOld(p, plane_hypotheses[up_near], cost_array[0], helper);
	}

	// down_near
	if (p.y < height - 1) {
		flag[2] = true;
		num_valid_pixels++;
		costMin = costs[down_near];
		costMinPoint = down_near;
		for (int i = 0; i < 3; ++i) {
			if (p.y < height - 2 - i && p.x > i) {
				int pointTemp = down_near + (1 + i) * width - (i + 1);
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.y < height - 2 - i && p.x < width - 1 - i) {
				int pointTemp = down_near + (1 + i) * width + (i + 1);
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		down_near = costMinPoint;
		ComputeMultiViewCostVectorOld(p, plane_hypotheses[down_near], cost_array[2], helper);
	}

	// left_near
	if (p.x > 0) {
		flag[4] = true;
		num_valid_pixels++;
		costMin = costs[left_near];
		costMinPoint = left_near;
		for (int i = 0; i < 3; ++i) {
			if (p.x > 1 + i && p.y > i) {
				int pointTemp = left_near - (1 + i) - (i + 1) * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.x > 1 + i && p.y < height - 1 - i) {
				int pointTemp = left_near - (1 + i) + (i + 1) * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		left_near = costMinPoint;
		ComputeMultiViewCostVectorOld(p, plane_hypotheses[left_near], cost_array[4], helper);
	}

	// right_near
	if (p.x < width - 1) {
		flag[6] = true;
		num_valid_pixels++;
		costMin = costs[right_near];
		costMinPoint = right_near;
		for (int i = 0; i < 3; ++i) {
			if (p.x < width - 2 - i && p.y > i) {
				int pointTemp = right_near + (1 + i) - (i + 1) * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			if (p.x < width - 2 - i && p.y < height - 1 - i) {
				int pointTemp = right_near + (1 + i) + (i + 1) * width;
				if (costs[pointTemp] < costMin) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
		}
		right_near = costMinPoint;
		ComputeMultiViewCostVectorOld(p, plane_hypotheses[right_near], cost_array[6], helper);
	}

	const int positions[8] = { up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far };

	// Multi-hypothesis Joint View Selection
	uchar* view_weights = &(helper->view_weight_cuda[center * MAX_IMAGES]);
	for (int i = 0; i < MAX_IMAGES; ++i) {
		view_weights[i] = 0;
	}
	float view_selection_priors[32] = { 0.0f };

	int neighbor_positions[4] = { center - width, center + width, center - 1, center + 1 };
	for (int i = 0; i < 4; ++i) {
		if (flag[2 * i]) {
			for (int j = 0; j < num_images - 1; ++j) {
				if (isSet(selected_views[neighbor_positions[i]], j) == 1) {
					view_selection_priors[j] += params->vs_prior_selected;
				}
				else {
					view_selection_priors[j] += params->vs_prior_unselected;
				}
			}
		}
	}

	float sampling_probs[32] = { 0.0f };
	float cost_threshold = params->vs_cost_thresh_init * expf((iter) * (iter) / (-params->vs_cost_thresh_decay));
	for (int i = 0; i < num_images - 1; i++) {
		float count = 0;
		int count_false = 0;
		float tmpw = 0;
		for (int j = 0; j < 8; j++) {
			if (cost_array[j][i] < cost_threshold) {
				tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-params->vs_good_sigma));
				count++;
			}
			if (cost_array[j][i] > params->vs_bad_cost) {
				count_false++;
			}
		}
		if (count > params->vs_min_good && count_false < params->vs_max_bad) {
			sampling_probs[i] = tmpw / count;
		}
		else if (count_false < params->vs_max_bad) {
			sampling_probs[i] = expf(cost_threshold * cost_threshold / (-params->vs_fallback_sigma));
		}
		sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
	}

	TransformPDFToCDF(sampling_probs, num_images - 1);
	for (int sample = 0; sample < params->vs_num_samples; ++sample) {
		const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

		for (int image_id = 0; image_id < num_images - 1; ++image_id) {
			const float prob = sampling_probs[image_id];
			if (prob > rand_prob) {
				view_weights[image_id] += 1;
				break;
			}
		}
	}

	unsigned int temp_selected_views = 0;
	int num_selected_view = 0;
	float weight_norm = 0;

	for (int i = 0; i < num_images - 1; ++i) {
		if (view_weights[i] > 0) {
			setBit(&temp_selected_views, i);
			weight_norm += view_weights[i];
			num_selected_view++;
		}
	}

	float final_costs[8] = { 0.0f };

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < num_images - 1; ++j) {
			if (view_weights[j] > 0) {
				final_costs[i] += view_weights[j] * cost_array[i][j];
			}
		}
		final_costs[i] /= weight_norm;
	}

	const int min_cost_idx = FindMinCostIndex(final_costs, 8);

	float cost_vector_now[32] = { 2.0f };
	ComputeMultiViewCostVectorOld(p, plane_hypotheses[center], cost_vector_now, helper);
	float cost_now = 0.0f;

	for (int i = 0; i < num_images - 1; ++i) {
		if (params->geom_consistency && params->use_impetus) {
			cost_now += view_weights[i] * (cost_vector_now[i] + params->geom_factor * ComputeGeomConsistencyCost(p, i + 1, plane_hypotheses[center], helper));
		}
		else {
			cost_now += view_weights[i] * cost_vector_now[i];
		}
	}
	cost_now /= weight_norm;
	costs[center] = cost_now;
	float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
	float4 plane_hypotheses_now = plane_hypotheses[center];

	if (flag[min_cost_idx]) {
		float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);

		// a neighbour's plane whose depth here leaves this pixel's bounds (view range or prior band) is rejected
		if (DepthInPixelBounds(helper, center, depth_before) && final_costs[min_cost_idx] < cost_now) {
			depth_now = depth_before;
			plane_hypotheses_now = plane_hypotheses[positions[min_cost_idx]];
			cost_now = final_costs[min_cost_idx];
			selected_views[center] = temp_selected_views;
		}
	}
	PlaneHypothesisRefinementStrong(&plane_hypotheses_now, &depth_now, &cost_now, &rand_states[center], view_weights, weight_norm, p, helper);

	if (params->state == REFINE_INIT) {
		if (cost_now < costs[center] - params->refine_init_margin) {
			costs[center] = cost_now;
			plane_hypotheses[center] = plane_hypotheses_now;
		}
	}
	else {
		costs[center] = cost_now;
		plane_hypotheses[center] = plane_hypotheses_now;
	}
}

__device__ void CheckerboardPropagationWeak(
	const int2 p,
	const int iter,
	DataPassHelper* helper
) {
	const int width = helper->width;
	const int height = helper->height;
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	float* costs = helper->costs_cuda;
	curandState* rand_states = helper->rand_states_cuda;
	unsigned int* selected_views = helper->selected_views_cuda;
	PatchMatchParams* params = helper->params;
	const Camera* cameras = helper->cameras_cuda;
	int num_images = params->num_images;

	if (p.x >= width || p.y >= height) {
		return;
	}

	const int center = p.y * width + p.x;

	// Adaptive Checkerboard Sampling
	float cost_array[8][32] = { 2.0f };
	bool flag[8] = { false };
	int num_valid_pixels = 0;

	int positions[8] = { 0 };
	float4 new_plane_hypothesis[8];

	for (int i = 0; i < 8; ++i) {
		const auto anchor_pt = GetAnchorPoint(p, i + 1, helper);
		if (anchor_pt.x == -1 || anchor_pt.y == -1 || helper->weak_info_cuda[anchor_pt.x + anchor_pt.y * width] != STRONG) {
			flag[i] = false;
			continue;
		}
		positions[i] = anchor_pt.x + anchor_pt.y * width;
		flag[i] = true;
		num_valid_pixels++;
		ComputeMultiViewCostVectorNew(p, plane_hypotheses[anchor_pt.x + anchor_pt.y * width], cost_array[i], helper);
		new_plane_hypothesis[i] = plane_hypotheses[anchor_pt.x + anchor_pt.y * width];
	}

	// Multi-hypothesis Joint View Selection
	uchar* view_weights = &(helper->view_weight_cuda[center * MAX_IMAGES]);
	for (int i = 0; i < MAX_IMAGES; ++i) {
		view_weights[i] = 0;
	}
	float view_selection_priors[32] = { 0.0f };
	for (int i = 0; i < 8; ++i) {
		const auto anchor_pt = GetAnchorPoint(p, i + 1, helper);
		if (anchor_pt.x == -1 || anchor_pt.y == -1) {
			continue;
		}
		for (int j = 0; j < num_images - 1; ++j) {
			if (isSet(selected_views[anchor_pt.x + anchor_pt.y * width], j) == 1) {
				view_selection_priors[j] += params->vs_prior_selected;
			}
			else {
				view_selection_priors[j] += params->vs_prior_unselected;
			}
		}
	}

	float sampling_probs[32] = { 0.0f };
	float cost_threshold = params->vs_cost_thresh_init * expf((iter) * (iter) / (-params->vs_cost_thresh_decay));
	for (int i = 0; i < num_images - 1; i++) {
		float count = 0;
		int count_false = 0;
		float tmpw = 0;
		for (int j = 0; j < 8; j++) {
			if (cost_array[j][i] < cost_threshold) {
				tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-params->vs_good_sigma));
				count++;
			}
			if (cost_array[j][i] > params->vs_bad_cost) {
				count_false++;
			}
		}
		if (count > params->vs_min_good && count_false < params->vs_max_bad) {
			sampling_probs[i] = tmpw / count;
		}
		else if (count_false < params->vs_max_bad) {
			sampling_probs[i] = expf(cost_threshold * cost_threshold / (-params->vs_fallback_sigma));
		}
		sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
	}

	TransformPDFToCDF(sampling_probs, num_images - 1);
	for (int sample = 0; sample < params->vs_num_samples; ++sample) {
		const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

		for (int image_id = 0; image_id < num_images - 1; ++image_id) {
			const float prob = sampling_probs[image_id];
			if (prob > rand_prob) {
				view_weights[image_id] += 1;
				break;
			}
		}
	}

	unsigned int temp_selected_views = 0;
	int num_selected_view = 0;
	float weight_norm = 0;

	for (int i = 0; i < num_images - 1; ++i) {
		if (view_weights[i] > 0) {
			setBit(&temp_selected_views, i);
			weight_norm += view_weights[i];
			num_selected_view++;
		}
	}

	float final_costs[8] = { 0.0f };

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < num_images - 1; ++j) {
			if (view_weights[j] > 0) {
				if (params->geom_consistency) {
					if (flag[i]) {
						final_costs[i] += view_weights[j] * (cost_array[i][j] + params->geom_factor * ComputeGeomConsistencyCost(p, j + 1, plane_hypotheses[positions[i]], helper));
					}
					else {
						final_costs[i] += view_weights[j] * (cost_array[i][j] + params->geom_factor * params->geom_max_cost);
					}
				}
				else {
					final_costs[i] += view_weights[j] * cost_array[i][j];
				}
			}
		}
		final_costs[i] /= weight_norm;
	}

	const int min_cost_idx = FindMinCostIndex(final_costs, 8);

	float cost_vector_now[32] = { 2.0f };
	ComputeMultiViewCostVectorNew(p, plane_hypotheses[center], cost_vector_now, helper);
	float cost_now = 0.0f;

	for (int i = 0; i < num_images - 1; ++i) {
		if (params->geom_consistency) {
			cost_now += view_weights[i] * (cost_vector_now[i] + params->geom_factor * ComputeGeomConsistencyCost(p, i + 1, plane_hypotheses[center], helper));
		}
		else {
			cost_now += view_weights[i] * cost_vector_now[i];
		}
	}
	cost_now /= weight_norm;
	costs[center] = cost_now;
	float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
	float4 plane_hypotheses_now = plane_hypotheses[center];

	if (flag[min_cost_idx]) {
		float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], new_plane_hypothesis[min_cost_idx], p);
		// a neighbour's plane whose depth here leaves this pixel's bounds (view range or prior band) is rejected
		if (DepthInPixelBounds(helper, center, depth_before) && final_costs[min_cost_idx] < cost_now) {
			depth_now = depth_before;
			plane_hypotheses_now = new_plane_hypothesis[min_cost_idx];
			cost_now = final_costs[min_cost_idx];
			selected_views[center] = temp_selected_views;
		}
	}
	PlaneHypothesisRefinementWeak(&plane_hypotheses_now, &depth_now, &cost_now, &rand_states[center], view_weights, weight_norm, p, helper);

	if (params->state == REFINE_INIT) {
		if (cost_now < costs[center] - params->refine_init_margin) {
			costs[center] = cost_now;
			plane_hypotheses[center] = plane_hypotheses_now;
		}
	}
	else {
		costs[center] = cost_now;
		plane_hypotheses[center] = plane_hypotheses_now;
	}
}

__global__ void BlackPixelUpdateWeak(const int iter, DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2;
	}
	else {
		p.y = p.y * 2 + 1;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}

	if (helper->weak_info_cuda[p.x + p.y * helper->width] == WEAK) {
		CheckerboardPropagationWeak(p, iter, helper);
	}
}

__global__ void RedPixelUpdateWeak(const int iter, DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2 + 1;
	}
	else {
		p.y = p.y * 2;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] == WEAK) {
		CheckerboardPropagationWeak(p, iter, helper);
	}
}

__global__ void BlackPixelUpdateStrong(const int iter, DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2;
	}
	else {
		p.y = p.y * 2 + 1;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] == WEAK) {
		return;
	}

	CheckerboardPropagationStrong(p, iter, helper);
}

__global__ void RedPixelUpdateStrong(const int iter, DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2 + 1;
	}
	else {
		p.y = p.y * 2;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] == WEAK) {
		return;
	}

	CheckerboardPropagationStrong(p, iter, helper);
}

__global__ void GetDepthandNormal(
	DataPassHelper* helper
) {
	const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	Camera* cameras = helper->cameras_cuda;
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	const int width = helper->width;;
	const int height = helper->height;

	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	plane_hypotheses[center].w = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
	plane_hypotheses[center] = TransformNormal(cameras[0], plane_hypotheses[center]);
}

__device__ void CheckerboardFilterStrong(
	const int2 p,
	DataPassHelper* helper
) {
	int width = helper->width;
	int height = helper->height;
	if (p.x >= width || p.y >= height) {
		return;
	}
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	float* costs = helper->costs_cuda;
	const int center = p.y * width + p.x;

	float filter[21];
	int index = 0;

	filter[index++] = plane_hypotheses[center].w;

	// Left
	const int left = center - 1;
	const int leftleft = center - 3;

	// Up
	const int up = center - width;
	const int upup = center - 3 * width;

	// Down
	const int down = center + width;
	const int downdown = center + 3 * width;

	// Right
	const int right = center + 1;
	const int rightright = center + 3;

	if (costs[center] < 0.001f) {
		return;
	}

	if (p.y > 0 && helper->weak_info_cuda[up] == STRONG) {
		filter[index++] = plane_hypotheses[up].w;
	}
	if (p.y > 2 && helper->weak_info_cuda[upup] == STRONG) {
		filter[index++] = plane_hypotheses[upup].w;
	}
	if (p.y > 4 && helper->weak_info_cuda[upup - width * 2] == STRONG) {
		filter[index++] = plane_hypotheses[upup - width * 2].w;
	}
	if (p.y < height - 1 && helper->weak_info_cuda[down] == STRONG) {
		filter[index++] = plane_hypotheses[down].w;
	}
	if (p.y < height - 3 && helper->weak_info_cuda[downdown] == STRONG) {
		filter[index++] = plane_hypotheses[downdown].w;
	}
	if (p.y < height - 5 && helper->weak_info_cuda[downdown + width * 2] == STRONG) {
		filter[index++] = plane_hypotheses[downdown + width * 2].w;
	}
	if (p.x > 0 && helper->weak_info_cuda[left] == STRONG) {
		filter[index++] = plane_hypotheses[left].w;
	}
	if (p.x > 2 && helper->weak_info_cuda[leftleft] == STRONG) {
		filter[index++] = plane_hypotheses[leftleft].w;
	}
	if (p.x > 4 && helper->weak_info_cuda[leftleft - 2] == STRONG) {
		filter[index++] = plane_hypotheses[leftleft - 2].w;
	}
	if (p.x < width - 1 && helper->weak_info_cuda[right] == STRONG) {
		filter[index++] = plane_hypotheses[right].w;
	}
	if (p.x < width - 3 && helper->weak_info_cuda[rightright] == STRONG) {
		filter[index++] = plane_hypotheses[rightright].w;
	}
	if (p.x < width - 5 && helper->weak_info_cuda[rightright + 2] == STRONG) {
		filter[index++] = plane_hypotheses[rightright + 2].w;
	}
	if (p.y > 0 && p.x < width - 2 && helper->weak_info_cuda[up + 2] == STRONG) {
		filter[index++] = plane_hypotheses[up + 2].w;
	}
	if (p.y < height - 1 && p.x < width - 2 && helper->weak_info_cuda[down + 2] == STRONG) {
		filter[index++] = plane_hypotheses[down + 2].w;
	}
	if (p.y > 0 && p.x > 1 && helper->weak_info_cuda[up - 2] == STRONG)
	{
		filter[index++] = plane_hypotheses[up - 2].w;
	}
	if (p.y < height - 1 && p.x>1 && helper->weak_info_cuda[down - 2] == STRONG) {
		filter[index++] = plane_hypotheses[down - 2].w;
	}
	if (p.x > 0 && p.y > 2 && helper->weak_info_cuda[left - width * 2] == STRONG)
	{
		filter[index++] = plane_hypotheses[left - width * 2].w;
	}
	if (p.x < width - 1 && p.y>2 && helper->weak_info_cuda[right - width * 2] == STRONG)
	{
		filter[index++] = plane_hypotheses[right - width * 2].w;
	}
	if (p.x > 0 && p.y < height - 2 && helper->weak_info_cuda[left + width * 2] == STRONG) {
		filter[index++] = plane_hypotheses[left + width * 2].w;
	}
	if (p.x < width - 1 && p.y < height - 2 && helper->weak_info_cuda[right + width * 2] == STRONG) {
		filter[index++] = plane_hypotheses[right + width * 2].w;
	}

	sort_small(filter, index);
	int median_index = index / 2;
	if (index % 2 == 0) {
		plane_hypotheses[center].w = (filter[median_index - 1] + filter[median_index]) / 2;
	}
	else {
		plane_hypotheses[center].w = filter[median_index];
	}
}

__global__ void BlackPixelFilterStrong(DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2;
	}
	else {
		p.y = p.y * 2 + 1;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] != WEAK) {
		CheckerboardFilterStrong(p, helper);
	}
}

__global__ void RedPixelFilterStrong(DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2 + 1;
	}
	else {
		p.y = p.y * 2;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] != WEAK) {
		CheckerboardFilterStrong(p, helper);
	}
}

__global__ void GenAnchors(
	DataPassHelper* helper
) {
	int width = helper->width;
	int height = helper->height;
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (point.x >= width || point.y >= height) {
		return;
	}
	const unsigned center = point.x + point.y * width;
	const uchar* weak_info = helper->weak_info_cuda;
	if (weak_info[center] != WEAK) {
		return;
	}
	const int min_margin = helper->params->border_margin;
	const float depth_diff = helper->params->depth_max - helper->params->depth_min;
	const int* anchors_map = helper->anchors_map_cuda;
	const PatchMatchParams* params = helper->params;
	const short2* weak_nearest_strong = helper->weak_nearest_strong;
	const Camera& camera = helper->cameras_cuda[0];
	const unsigned offset = anchors_map[center] * ANCHOR_NUM;
	const float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	curandState* rand_state = &(helper->rand_states_cuda[center]);
	short2* anchors = &(helper->anchors_cuda[offset]);
	uchar* weak_reliable = &(helper->weak_reliable_cuda[center]);
	// init for invalid points
	for (int i = 0; i < ANCHOR_NUM; ++i) {
		anchors[i].x = -1;
		anchors[i].y = -1;
	}
	anchors[0] = make_short2(point.x, point.y); // the first point is the center point
	short2 strong_points[8 * 4];
	bool dir_valid[8 * 4];
	for (int i = 0; i < 32; ++i) {
		strong_points[i] = make_short2(-1, -1);
		dir_valid[i] = false;
	}
	int origin_direction_index = -1;
	int strong_point_size = 0;
	const int rotate_time = params->rotate_time; // max is 4 from [1, 2, 4]
	const float angle = 45.0f / rotate_time;
	const float cos_angle = cos(angle * M_PI / 180.f);
	const float sin_angle = sin(angle * M_PI / 180.f);
	const float threshhold = cos((angle / 2.0f) * M_PI / 180.0f);
	const int shift_range = MAX((int)(tan((angle / 2.0f) * M_PI / 180.0f) * 20), 1);
	const float ransac_threshold = params->ransac_threshold;

	for (int origin_direction_x = -1; origin_direction_x <= 1; ++origin_direction_x) {
		for (int origin_direction_y = -1; origin_direction_y <= 1; ++origin_direction_y) {
			if (origin_direction_x == 0 && origin_direction_y == 0) {
				continue;
			}
			float2 origin_direction = make_float2(origin_direction_x, origin_direction_y);
			NormalizeVec2(&origin_direction);
			origin_direction_index++;
			for (int rotate_iter = 0; rotate_iter < rotate_time; ++rotate_iter) {
				int dir_index = origin_direction_index * 4 + rotate_iter;
                bool stop_expand = false;
				for (int radius = 2; radius <= MAX_SEARCH_RADIUS; radius = MIN(radius * 2, radius + 25)) {
					float2 test_pt = make_float2(point.x + origin_direction.x * radius, point.y + origin_direction.y * radius);
					if (test_pt.x < 0 || test_pt.y < 0 || test_pt.x >= width || test_pt.y >= height) {
						break;
					}
					for (int radius_iter = 0; radius_iter < 4; ++radius_iter) {
						int rand_x_shift = (curand(rand_state) % 2 == 0 ? 1 : -1) * curand(rand_state) % shift_range;
						int rand_y_shift = (curand(rand_state) % 2 == 0 ? 1 : -1) * curand(rand_state) % shift_range;
						float2 direction = make_float2(origin_direction.x * 20 + rand_x_shift, origin_direction.y * 20 + rand_y_shift);
						NormalizeVec2(&direction);
						short2 anchor_pt = make_short2(point.x + direction.x * radius, point.y + direction.y * radius);
						if (anchor_pt.x < min_margin || anchor_pt.y < min_margin || anchor_pt.x >= width - min_margin || anchor_pt.y >= height - min_margin) {
							continue;
						}
						int anchor_pt_center = anchor_pt.x + anchor_pt.y * width;
                        anchor_pt = weak_nearest_strong[anchor_pt_center];
                        if (anchor_pt.x == -1 || anchor_pt.y == -1) {
                            continue;
                        }
                        anchor_pt_center = anchor_pt.x + anchor_pt.y * width;

						float2 test_direction = make_float2(anchor_pt.x - point.x, anchor_pt.y - point.y);
						NormalizeVec2(&test_direction);
						float cos_angle = Vec2DotVec2(test_direction, origin_direction);
						if (cos_angle > threshhold) {
							strong_points[dir_index] = anchor_pt;
							dir_valid[dir_index] = true;
							strong_point_size++;
							break;
						}
					}
					if (dir_valid[dir_index]) {
						break;
					}
                    if (stop_expand) {
                        break;
                    }
				}
				// rotate
				{
					float2 rotated_direction;
					rotated_direction.x = origin_direction.x * cos_angle - origin_direction.y * sin_angle;
					rotated_direction.y = origin_direction.x * sin_angle + origin_direction.y * cos_angle;
					NormalizeVec2(&rotated_direction);
					origin_direction = rotated_direction;
				}
			}
		}
	}

	if (strong_point_size <= 3) {
		*weak_reliable = 0;
		return;
	}
	float4 best_plane;
	int use_a_index = -1, use_b_index = -1, use_c_index = -1;
	bool has_valid_plane = false;
	short2 strong_points_valid[8 * 4];
	float3 strong_points_valid_3d[8 * 4];
	int valid_count = 0;
	float X[3];
	Get3DPoint(camera, point, plane_hypotheses[center].w, X);
	float3 center_point_world = make_float3(X[0], X[1], X[2]);
	for (int i = 0; i < 32; ++i) {
		strong_points_valid[i] = make_short2(-1, -1);
		if (dir_valid[i]) {
			const auto& strong_point = strong_points[i];
			int strong_point_center = strong_point.x + strong_point.y * width;
			strong_points_valid[valid_count] = strong_points[i];
			Get3DPoint(camera, strong_point, plane_hypotheses[strong_point_center].w, X);
			strong_points_valid_3d[valid_count] = make_float3(X[0], X[1], X[2]);
			valid_count++;
		}
	}
	{	// RANSAC to find a good plane
		int iteration = params->anchor_ransac_iters;
		float min_cost = FLT_MAX;
		int max_count = 3;
		while (iteration--) {
			int a_index = curand(rand_state) % valid_count;
			int b_index = curand(rand_state) % valid_count;
			int c_index = curand(rand_state) % valid_count;
			if (a_index == b_index || b_index == c_index || a_index == c_index) {
				continue;
			}
			// compute triangle
			if (!PointinTriangle(strong_points_valid[a_index], strong_points_valid[b_index], strong_points_valid[c_index], point)) {
				continue;
			}
			const float3& A = strong_points_valid_3d[a_index];
			const float3& B = strong_points_valid_3d[b_index];
			const float3& C = strong_points_valid_3d[c_index];
			float3 A_C = make_float3(A.x - C.x, A.y - C.y, A.z - C.z);
			float3 B_C = make_float3(B.x - C.x, B.y - C.y, B.z - C.z);
			float4 cross_vec;
			cross_vec.x = A_C.y * B_C.z - B_C.y * A_C.z;
			cross_vec.y = -(A_C.x * B_C.z - B_C.x * A_C.z);
			cross_vec.z = A_C.x * B_C.y - B_C.x * A_C.y;
			if ((cross_vec.x == 0 && cross_vec.y == 0 && cross_vec.z == 0) || isnan(cross_vec.x) || isnan(cross_vec.y) || isnan(cross_vec.z)) {
				continue;
			}
			NormalizeVec3(&cross_vec);
			cross_vec.w = -(cross_vec.x * A.x + cross_vec.y * A.y + cross_vec.z * A.z);
			int temp_count = 0;
			float strong_dist = 0.0f;
			for (int strong_index = 0; strong_index < valid_count; ++strong_index) {
				const float3& temp_point = strong_points_valid_3d[strong_index];
				float distance = fabs(cross_vec.x * temp_point.x + cross_vec.y * temp_point.y + cross_vec.z * temp_point.z + cross_vec.w);
				if (distance / depth_diff < ransac_threshold) {
					temp_count++;
					strong_dist += distance;
				}
			}
			if (temp_count < params->anchor_min_inliers) {
				continue;
			}
			if (temp_count > max_count) {
				max_count = temp_count;
				const float center_distance = fabs(cross_vec.x * center_point_world.x + cross_vec.y * center_point_world.y + cross_vec.z * center_point_world.z + cross_vec.w);
				strong_dist /= temp_count;
				min_cost = center_distance;
				best_plane = cross_vec;
				has_valid_plane = true;
				use_a_index = a_index;
				use_b_index = b_index;
				use_c_index = c_index;
			}
			else if (temp_count == max_count) {
				const float center_distance = fabs(cross_vec.x * center_point_world.x + cross_vec.y * center_point_world.y + cross_vec.z * center_point_world.z + cross_vec.w);
				if (center_distance < min_cost) {
					max_count = temp_count;
					strong_dist /= temp_count;
					min_cost = center_distance;
					best_plane = cross_vec;
					use_a_index = a_index;
					use_b_index = b_index;
					use_c_index = c_index;
				}
			}
		}
	}
	if (!has_valid_plane) {
		*weak_reliable = 0;
		return;
	}
	float weight[32];
	for (int i = 0; i < valid_count; ++i) {
		const float3& temp_point = strong_points_valid_3d[i];
		float distance = fabs(best_plane.x * temp_point.x + best_plane.y * temp_point.y + best_plane.z * temp_point.z + best_plane.w);
		if (distance / depth_diff >= ransac_threshold) {
			strong_points_valid[i] = make_short2(-1, -1);
			weight[i] = FLT_MAX;
			continue;
		}
		if (i == use_a_index || i == use_b_index || i == use_c_index) {
			distance -= 1;
		}
		weight[i] = distance;
	}
	sort_small_weighted(strong_points_valid, weight, valid_count);
	for (int i = 1; i < ANCHOR_NUM; ++i) {
		anchors[i] = strong_points_valid[i - 1];
        if (anchors[i].x == -1 || anchors[i].y == -1) {
            continue;
        }
	}
	*weak_reliable = 1;
}

__global__ void NeigbourUpdate(
	DataPassHelper* helper
) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}
	const int center = point.x + point.y * width;
	if (helper->weak_info_cuda[center] != WEAK) {
		return;
	}
	if (helper->weak_reliable_cuda[center] != 1) {
		helper->weak_info_cuda[center] = UNKNOWN;
	}
}


__global__ void DepthToWeak(DataPassHelper *helper, float *reliable_curve) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}

	const int min_margin = helper->params->border_margin;
	const int center = point.x + point.y * width;

	if (point.x < min_margin || point.y < min_margin || point.x >= width - min_margin || point.y >= height - min_margin) {
		helper->weak_info_cuda[center] = UNKNOWN;
		return;
	}
	const auto &image = helper->texture_objects_cuda[0].images[0];
	const float center_pix = tex2D<float>(image, point.x + 0.5f, point.y + 0.5f);

	const Camera *cameras = helper->cameras_cuda;
	const unsigned *selected_views = helper->selected_views_cuda;
	const int num_images = helper->params->num_images;
	const uchar *view_weight = &(helper->view_weight_cuda[MAX_IMAGES * center]);
	float4 origin_plane_hypothesis;
	origin_plane_hypothesis = helper->plane_hypotheses_cuda[center];
	origin_plane_hypothesis = TransformNormal2RefCam(cameras[0], origin_plane_hypothesis);
	float origin_depth = origin_plane_hypothesis.w;
	if (origin_depth == 0) {
		helper->weak_info_cuda[center] = UNKNOWN;
		return;
	}
	// compute cost now and baseline

	float base_line = 0;
	int valid_src = 0;
	float weight_normal = 0.0f;
	for (int src_index = 1; src_index < num_images; ++src_index) {
		int view_index = src_index - 1;
		if (isSet(selected_views[center], view_index)) {
			weight_normal += view_weight[view_index];
			float c_dist[3];
			c_dist[0] = cameras[0].c[0] - cameras[src_index].c[0];
			c_dist[1] = cameras[0].c[1] - cameras[src_index].c[1];
			c_dist[2] = cameras[0].c[2] - cameras[src_index].c[2];
			double temp_val = c_dist[0] * c_dist[0] + c_dist[1] * c_dist[1] + c_dist[2] * c_dist[2];
			base_line += sqrtf(temp_val);
			valid_src++;
		}
	}
	if (valid_src == 0) {
		helper->weak_info_cuda[center] = UNKNOWN;
		return;
	}
	base_line /= valid_src;

	float disp = cameras[0].K[0] * base_line / origin_depth;
	const int radius = 30;
	const int p_costs_size = 2 * radius + 1;
	float p_costs[p_costs_size];
	int increment = 1;

	for (int p_disp = -radius * increment; p_disp <= radius * increment; p_disp += increment) {

		float p_depth = cameras[0].K[0] * base_line / (disp + p_disp);
		if (!DepthInPixelBounds(helper, center, p_depth)) {   // view range or prior band
			p_costs[p_disp + radius] = 2.0f;
			continue;
		}
		float4 temp_plane_hypothesis = origin_plane_hypothesis;
		temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], point, p_depth, temp_plane_hypothesis);
		float p_cost = 0.0f;
		for (int src_index = 1; src_index < num_images; ++src_index) {
			int view_index = src_index - 1;
			float temp_cost = 0.0f;
 			if (isSet(selected_views[center], view_index)) {
				temp_cost += ComputeBilateralNCCOld(point, src_index, temp_plane_hypothesis, helper);
				if (helper->params->geom_consistency) {
					temp_cost += helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_plane_hypothesis, helper);
				}
				p_cost += (temp_cost * view_weight[view_index]);
			}
		}
		p_cost /= weight_normal;
		p_costs[p_disp + radius] = MIN(2.0f, p_cost);
	}

	if (reliable_curve != nullptr) {
		if (p_costs_size != RELIABLE_CURVE_SAMPLE_NUM) {
			printf("Export reliable curve error: p_costs_size != RELIABLE_CURVE_SAMPLE_NUM\n");

		} else {
			float *reliable_curve_cur = &(reliable_curve[center * p_costs_size]);
			for (int i = 0; i < p_costs_size; ++i) {
				reliable_curve_cur[i] = p_costs[i];
			}
		}
	}

	// find peaks
	bool is_peak[p_costs_size];
	for (int i = 0; i < p_costs_size; ++i) {
		is_peak[i] = false;
	}

	int peak_count = 0;
	int min_peak = 0;
	float min_cost = 2.0f;
	for (int i = 2; i < p_costs_size - 2; ++i) {
		if (p_costs[i - 1] > p_costs[i] && p_costs[i + 1] > p_costs[i]) {
			is_peak[i] = true;
			peak_count++;
			if (p_costs[i] < min_cost) {
				min_peak = i;
				min_cost = p_costs[i];
			}
		}
	}

	if (abs(min_peak - radius) > helper->params->weak_peak_radius || p_costs[min_peak] > helper->params->weak_max_peak_cost) {
		helper->weak_info_cuda[center] = WEAK;
		return;
	}

	if (peak_count == 1) {
		if (p_costs[min_peak] <= helper->params->strong_single_peak_cost) {
			helper->weak_info_cuda[center] = STRONG;
		}
		else {
			helper->weak_info_cuda[center] = WEAK;
		}
		return;
	}

	float var = 0.0f;
	for (int i = 2; i < p_costs_size - 2; ++i) {
		if (is_peak[i] && i != min_peak) {
			float dist = p_costs[i] - min_cost;
			var += dist * dist;
		}
	}
	var = sqrtf(var);
	var /= (peak_count - 1);

	if (var > helper->params->strong_multi_peak_var) {
		helper->weak_info_cuda[center] = STRONG;
	} else {
		helper->weak_info_cuda[center] = WEAK;
	}
}

__global__ void WeakFilter(DataPassHelper *helper, uchar *weak_info_copy) {
    const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = helper->width;
    const int height = helper->height;
    if (point.x >= width || point.y >= height) {
        return;
    }
    const int center = point.x + point.y * width;
    if (helper->weak_info_cuda[center] != STRONG) {
        return;
    }
    const int radius = helper->params->weak_filter_radius;
    for (int x = - radius; x <= radius; ++x) {
        for (int y = - radius; y <= radius; ++y) {
            if (x == 0 && y == 0) {
                continue;
            }
            int2 neighbor = make_int2(point.x + x, point.y + y);
            if (neighbor.x < 0 || neighbor.x >= width || neighbor.y < 0 || neighbor.y >= height) {
                continue;
            }
            int neighbor_center = neighbor.x + neighbor.y * width;
            if (helper->weak_info_cuda[neighbor_center] == STRONG) {
                return;
            }
        }
    }
    weak_info_copy[center] = UNKNOWN;
}

__global__ void ConfidenceCompute(DataPassHelper *helper) {
    const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = helper->width;
    const int height = helper->height;
    if (point.x >= width || point.y >= height) {
        return;
    }
    const int center = point.x + point.y * width;
    uchar *confidence = helper->confidence_cuda;
    confidence[center] = 0;

    const Camera& ref_camera = helper->cameras_cuda[0];
    const auto *params = helper->params;
    const auto selected_view = helper->selected_views_cuda[center];
    const auto ref_depth = helper->plane_hypotheses_cuda[center].w;

    if (ref_depth <= 0.0f) {
        helper->weak_info_cuda[center] = UNKNOWN;
        return;
    }
    const float3 forward_point = Get3DPointonWorld_cu(point.x, point.y, ref_depth, ref_camera);
    int num_consistence = 1; // init with 1
    const int exist_in_src_weight = params->conf_w_exist;
    const int reproj_pixel_weight = params->conf_w_reproj;
    const int reproj_depth_weight = params->conf_w_depth;

    for (int i = 0; i < params->num_images - 1; ++i) {
        if (!isSet(selected_view, i)) {
            continue;
        }

        int src_idx = i + 1;
        const Camera& src_camera = helper->cameras_cuda[src_idx];
        const cudaTextureObject_t depth_image = helper->texture_depths_cuda[0].images[src_idx];
        float2 src_pt;
        float src_d;
        ProjectonCamera_cu(forward_point, src_camera, src_pt, src_d);
        const auto src_depth = tex2D<float>(depth_image, (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);
        if (src_depth <= 0.0f) {
            continue;
        }

        num_consistence += exist_in_src_weight;
        float3 src_3D_pt = Get3DPointonWorld_cu(src_pt.x, src_pt.y, src_depth, src_camera);
        float2 backward_point;
        float ref_d;
        ProjectonCamera_cu(src_3D_pt, ref_camera, backward_point, ref_d);
        const float diff_col = point.x - backward_point.x;
        const float diff_row = point.y - backward_point.y;
        const float pixel_diff = sqrtf(diff_col * diff_col + diff_row * diff_row);
        if (pixel_diff <= params->conf_reproj_px) {
            num_consistence += reproj_pixel_weight;
        }
        const float relative_depth_diff = fabsf(ref_depth - ref_d) / ref_depth;
        if (relative_depth_diff <= params->conf_depth_rel) {
            num_consistence += reproj_depth_weight;
        }
    }
    if (num_consistence > 255) {
        num_consistence = 255;
    }
    confidence[center] = (uchar)num_consistence;
}

__global__ void LocalRefine(DataPassHelper* helper) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}

	const int center = point.x + point.y * width;

	const Camera* cameras = helper->cameras_cuda;
	const unsigned* selected_views = helper->selected_views_cuda;
	const int num_images = helper->params->num_images;
	const uchar* view_weight = &(helper->view_weight_cuda[MAX_IMAGES * center]);
	float4 origin_plane_hypothesis;
	origin_plane_hypothesis = helper->plane_hypotheses_cuda[center];
	origin_plane_hypothesis = TransformNormal2RefCam(cameras[0], origin_plane_hypothesis);
	float origin_depth = origin_plane_hypothesis.w;
	if (origin_depth == 0) {
		return;
	}

	// compute cost now and baseline
	float cost_now = 0.0f;
	float base_line = 0;
	int valid_src = 0;
	float weight_normal = 0.0f;
	for (int src_index = 1; src_index < num_images; ++src_index) {
		int view_index = src_index - 1;
		if (isSet(selected_views[center], view_index)) {
			float4 temp_plane_hypothesis = origin_plane_hypothesis;
			temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], point, origin_depth, temp_plane_hypothesis);
			float temp_cost = ComputeBilateralNCCOld(point, src_index, temp_plane_hypothesis, helper);
			if (helper->params->geom_consistency) {
				temp_cost += helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_plane_hypothesis, helper);
			}
			cost_now += (temp_cost * view_weight[view_index]);
			weight_normal += view_weight[view_index];
			float c_dist[3];
			c_dist[0] = cameras[0].c[0] - cameras[src_index].c[0];
			c_dist[1] = cameras[0].c[1] - cameras[src_index].c[1];
			c_dist[2] = cameras[0].c[2] - cameras[src_index].c[2];
			double temp_val = c_dist[0] * c_dist[0] + c_dist[1] * c_dist[1] + c_dist[2] * c_dist[2];
			base_line += sqrtf(temp_val);
			valid_src++;
		}
	}

	if (weight_normal == 0 || valid_src == 0) {
		return;
	}

	cost_now /= weight_normal;
	base_line /= valid_src;

	float disp = cameras[0].K[0] * base_line / origin_depth;
	const int radius = helper->params->local_refine_radius;

	float min_cost = 2.0f;
	float best_depth = origin_depth;
	for (int p_disp = -radius; p_disp <= radius; ++p_disp) {
		float p_depth = cameras[0].K[0] * base_line / (disp + p_disp);
		if (!DepthInPixelBounds(helper, center, p_depth)) {   // view range or prior band
			continue;
		}
		float4 temp_plane_hypothesis = origin_plane_hypothesis;
		temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], point, p_depth, temp_plane_hypothesis);
		float temp_cost = 0.0f;
		for (int src_index = 1; src_index < num_images; ++src_index) {
			int view_index = src_index - 1;
			if (isSet(selected_views[center], view_index)) {
				temp_cost += (ComputeBilateralNCCOld(point, src_index, temp_plane_hypothesis, helper) * view_weight[view_index]);
				if (helper->params->geom_consistency) {
					temp_cost += (helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_plane_hypothesis, helper) * view_weight[view_index]);
				}
			}
		}
		temp_cost /= weight_normal;
		if (temp_cost < min_cost) {
			min_cost = temp_cost;
			best_depth = p_depth;
		}
	}
	if (cost_now - min_cost > helper->params->local_refine_margin) {
		helper->plane_hypotheses_cuda[center].w = best_depth;
	}
}

__global__ void FindNearestStrongPoint(DataPassHelper* helper) {
    const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = helper->width;
    const int height = helper->height;
    if (point.x >= width || point.y >= height) {
        return;
    }
    const uchar* weak_info = helper->weak_info_cuda;
    const uchar* confidence = helper->confidence_cuda;
    short2* weak_nearest_strong = helper->weak_nearest_strong;
    const int center = point.x + point.y * width;
    weak_nearest_strong[center].x = -1;
    weak_nearest_strong[center].y = -1;
    const uchar center_confidence = confidence[center];

    if (weak_info[center] == WEAK || weak_info[center] == UNKNOWN) {
        uchar best_confidence = 0;
        short2 best_point = make_short2(-1, -1);
        float min_dist = FLT_MAX;
        const int radius = helper->params->nearest_strong_radius;
        for (int x = -radius; x <= radius; ++x) {
            for (int y = -radius; y <= radius; ++y) {
                short2 temp_point = make_short2(point.x + x, point.y + y);
                if (temp_point.x < 0 || temp_point.x >= width || temp_point.y < 0 || temp_point.y >= height) {
                    continue;
                }
                int temp_center = temp_point.x + temp_point.y * width;
                if (weak_info[temp_center] != STRONG) {
                    continue;
                }
                if (confidence[temp_center] < center_confidence) {
                    continue;
                }
                float temp_dist = sqrtf(x * x + y * y);
                if (temp_dist < min_dist) {
                    min_dist = temp_dist;
                    best_point = temp_point;
                    best_confidence = confidence[temp_center];
                } else if (temp_dist == min_dist) {
                    if (confidence[temp_center] > best_confidence) {
                        best_point = temp_point;
                        best_confidence = confidence[temp_center];
                    }
                }
            }
        }
        weak_nearest_strong[center] = best_point;
    } else if (weak_info[center] == STRONG) {
        weak_nearest_strong[center] = make_short2(point.x, point.y);
    }
}

__global__ void RANSACToGetFitPlane(DataPassHelper* helper) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}
	const uchar* weak_info = helper->weak_info_cuda;
	const int center = point.x + point.y * width;
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	float4* fit_plane_hypothese = helper->fit_plane_hypotheses_cuda;
	if (weak_info[center] != WEAK) {
		fit_plane_hypothese[center] = plane_hypotheses[center];
		return;
	}
	// make sure that the plane is in the ref camera coord
	curandState* rand_state = &(helper->rand_states_cuda[center]);
	const auto& camera = helper->cameras_cuda[0];

	short2 strong_points[ANCHOR_NUM - 1];
	float3 strong_points_3d[ANCHOR_NUM - 1];
	int strong_count = 0;
	float X[3];
	for (int i = 1; i < ANCHOR_NUM; ++i) {
		short2 temp_point = GetAnchorPoint(point, i, helper);
		if (temp_point.x == -1 || temp_point.y == -1) {
			continue;
		}
		strong_points[strong_count].x = temp_point.x;
		strong_points[strong_count].y = temp_point.y;
		// get 3d point in ref camera coord
		const int temp_center = temp_point.x + temp_point.y * width;
		float depth = ComputeDepthfromPlaneHypothesis(camera, plane_hypotheses[temp_center], make_int2(temp_point.x, temp_point.y));
		Get3DPoint(camera, strong_points[strong_count], depth, X);
		strong_points_3d[strong_count].x = X[0];
		strong_points_3d[strong_count].y = X[1];
		strong_points_3d[strong_count].z = X[2];
		strong_count++;
	}
	if (strong_count < 3) {
		fit_plane_hypothese[center] = plane_hypotheses[center];
		return;
	}

	int iteration = helper->params->fit_ransac_iters;
	float min_cost = FLT_MAX;
	float4 best_plane;
	bool has_best_plane = false;
	while (iteration--)
	{
		int a_index = curand(rand_state) % strong_count;
		int b_index = curand(rand_state) % strong_count;
		int c_index = curand(rand_state) % strong_count;

		if (a_index == b_index || b_index == c_index || a_index == c_index) {
			continue;
		}

		if (!PointinTriangle(strong_points[a_index], strong_points[b_index], strong_points[c_index], point)) {
			continue;
		}

		const float3& A = strong_points_3d[a_index];
		const float3& B = strong_points_3d[b_index];
		const float3& C = strong_points_3d[c_index];

		float3 A_C = make_float3(A.x - C.x, A.y - C.y, A.z - C.z);
		float3 B_C = make_float3(B.x - C.x, B.y - C.y, B.z - C.z);

		float4 cross_vec;
		cross_vec.x = A_C.y * B_C.z - B_C.y * A_C.z;
		cross_vec.y = -(A_C.x * B_C.z - B_C.x * A_C.z);
		cross_vec.z = A_C.x * B_C.y - B_C.x * A_C.y;
		if ((cross_vec.x == 0 && cross_vec.y == 0 && cross_vec.z == 0) || isnan(cross_vec.x) || isnan(cross_vec.y) || isnan(cross_vec.z)) {
			continue;
		}
		NormalizeVec3(&cross_vec);
		cross_vec.w = -(cross_vec.x * A.x + cross_vec.y * A.y + cross_vec.z * A.z);
		float temp_cost = 0.0f;
		for (int strong_index = 0; strong_index < strong_count; ++strong_index) {
			if (strong_index == a_index || strong_index == b_index || strong_index == c_index) {
				continue;
			}
			const float3& temp_point = strong_points_3d[strong_index];
			float distance = fabs(cross_vec.x * temp_point.x + cross_vec.y * temp_point.y + cross_vec.z * temp_point.z + cross_vec.w);
			temp_cost += distance;
		}
		if (temp_cost < min_cost) {
			min_cost = temp_cost;
			best_plane = cross_vec;
			has_best_plane = true;
		}
		if (min_cost == 0) {
			break;
		}
	}
	if (has_best_plane) {
		float depth = ComputeDepthfromPlaneHypothesis(camera, plane_hypotheses[center], point);
		float4 view_direction = GetViewDirection(camera, point, depth);
		float dot_product = best_plane.x * view_direction.x + best_plane.y * view_direction.y + best_plane.z * view_direction.z;
		if (dot_product > 0) {
			best_plane.x = -best_plane.x;
			best_plane.y = -best_plane.y;
			best_plane.z = -best_plane.z;
			best_plane.w = -best_plane.w;
		}

		fit_plane_hypothese[center] = best_plane;
	}
	else {
		fit_plane_hypothese[center] = make_float4(0, 0, 0, 0);
	}
}

void APD::ExportFitNormal() {
    std::shared_ptr<float4[]> fit_plane_hypotheses_host(new float4[width * height]);
    cudaMemcpy(fit_plane_hypotheses_host.get(), fit_plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
    cv::Mat normal_fit = cv::Mat::zeros(height, width, CV_32FC3);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float4 plane = fit_plane_hypotheses_host[r * width + c];
            normal_fit.at<cv::Vec3f>(r, c) = cv::Vec3f(plane.x, plane.y, plane.z);
        }
    }
    path normal_img_path = problem.result_folder / path("fit_normal_" + std::to_string(problem.iteration) + ".png");
    ShowNormalMap(normal_img_path, normal_fit);
}

void APD::ExportAnchors() {
    path anchors_map_path = problem.result_folder / path("anchors_map.bin");
    path anchors_path = problem.result_folder / path("anchors.bin");
    WriteBinMat(anchors_map_path, anchors_map_host);
    std::shared_ptr<short2 []> anchors_host(new short2[weak_count * ANCHOR_NUM]);
    cudaMemcpy(anchors_host.get(), anchors_cuda, sizeof(short2) * weak_count * ANCHOR_NUM, cudaMemcpyDeviceToHost);
    ofstream out(anchors_path, std::ios_base::binary);
    int anchor_sample_num = ANCHOR_NUM;
    out.write((char*)&weak_count, sizeof(int));
    out.write((char*)&anchor_sample_num, sizeof(int));
    out.write((char*)anchors_host.get(), sizeof(short2) * weak_count * ANCHOR_NUM);
    out.close();
}

void APD::ExportNearestStrong() {
    std::shared_ptr<short2[]> nearest_strong_host(new short2[width * height]);
    cudaMemcpy(nearest_strong_host.get(), weak_nearest_strong_cuda, sizeof(short2) * width * height, cudaMemcpyDeviceToHost);
    cv::Mat nearest_strong_map = cv::Mat::zeros(height, width, CV_8UC3);
    std::unordered_map<int, uchar3> color_map;
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            short2 nearest_strong = nearest_strong_host[r * width + c];
            int center = nearest_strong.y * width + nearest_strong.x;
            if (color_map.find(center) != color_map.end()) {
                nearest_strong_map.at<cv::Vec3b>(r, c) = cv::Vec3b(color_map[center].x, color_map[center].y, color_map[center].z);
            }
            else {
                uchar3 color = make_uchar3(rand() % 255, rand() % 255, rand() % 255);
                color_map[center] = color;
                nearest_strong_map.at<cv::Vec3b>(r, c) = cv::Vec3b(color.x, color.y, color.z);
            }
        }
    }
    path nearest_strong_img_path = problem.result_folder / path("nearest_strong_" + std::to_string(problem.iteration) + ".png");
    cv::imwrite(nearest_strong_img_path.string(), nearest_strong_map);
}

void APD::ExportReliableCurve(std::shared_ptr<float []> reliable_curve) {
	path reliable_curve_path = problem.result_folder / path("reliable_curve.bin");
	ofstream out(reliable_curve_path, std::ios_base::binary);
	// write width and height
	out.write((char*)&width, sizeof(int));
	out.write((char*)&height, sizeof(int));
	const int num_sample = RELIABLE_CURVE_SAMPLE_NUM;
	out.write((char*)&num_sample, sizeof(int));
	out.write((char*)reliable_curve.get(), sizeof(float) * width * height * RELIABLE_CURVE_SAMPLE_NUM);
	out.close();
}

void APD::RunPatchMatch() {
	int BLOCK_W = 32;
	int BLOCK_H = (BLOCK_W / 2);

	dim3 grid_size_full;
	grid_size_full.x = (width + 16 - 1) / 16;
	grid_size_full.y = (height + 16 - 1) / 16;
	grid_size_full.z = 1;
	dim3 block_size_full;
	block_size_full.x = 16;
	block_size_full.y = 16;
	block_size_full.z = 1;

	dim3 grid_size_half;
	grid_size_half.x = (width + BLOCK_W - 1) / BLOCK_W;
	grid_size_half.y = ((height / 2) + BLOCK_H - 1) / BLOCK_H;
	grid_size_half.z = 1;
	dim3 block_size_half;
	block_size_half.x = BLOCK_W;
	block_size_half.y = BLOCK_H;
	block_size_half.z = 1;

	InitRandomStates << <grid_size_full, block_size_full >> > (helper_cuda);
	if (problem.params.use_APD) {
		FindNearestStrongPoint << <grid_size_full, block_size_full >> > (helper_cuda);
        GenAnchors << <grid_size_full, block_size_full >> > (helper_cuda);
		NeigbourUpdate << <grid_size_full, block_size_full >> > (helper_cuda);
	}

    if (problem.export_anchor) {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        ExportAnchors();
    }

	RandomInitialization << <grid_size_full, block_size_full >> > (helper_cuda);

	for (int i = 0; i < params_host.max_iterations; ++i) {
		BlackPixelUpdateStrong << <grid_size_half, block_size_half >> > (i, helper_cuda);
		RedPixelUpdateStrong << <grid_size_half, block_size_half >> > (i, helper_cuda);

		if (problem.params.use_APD) {
			RANSACToGetFitPlane << <grid_size_full, block_size_full >> > (helper_cuda);
			BlackPixelUpdateWeak << <grid_size_half, block_size_half >> > (i, helper_cuda);
			RedPixelUpdateWeak << <grid_size_half, block_size_half >> > (i, helper_cuda);
		}
	}

	GetDepthandNormal << <grid_size_full, block_size_full >> > (helper_cuda);
	if (params_host.median_filter) {
		BlackPixelFilterStrong << <grid_size_half, block_size_half >> > (helper_cuda);
		RedPixelFilterStrong << <grid_size_half, block_size_half >> > (helper_cuda);
	}
	float *reliable_curve_cuda = nullptr;
	if (problem.export_reliable_curve) {
		cudaMalloc((void**)(&reliable_curve_cuda), sizeof(float) * width * height * RELIABLE_CURVE_SAMPLE_NUM);
	}
    DepthToWeak << <grid_size_full, block_size_full >> > (helper_cuda, reliable_curve_cuda);
	if (problem.export_reliable_curve) {
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		std::shared_ptr<float []> reliable_curve_host(new float[width * height * RELIABLE_CURVE_SAMPLE_NUM]);
		cudaMemcpy(reliable_curve_host.get(), reliable_curve_cuda, sizeof(float) * width * height * RELIABLE_CURVE_SAMPLE_NUM, cudaMemcpyDeviceToHost);
		ExportReliableCurve(reliable_curve_host);
		cudaFree(reliable_curve_cuda);
	}

    if (problem.params.geom_consistency || problem.params.use_APD) {
        ConfidenceCompute << < grid_size_full, block_size_full >> > (helper_cuda);
    }
    if (params_host.local_refine) {
        LocalRefine << <grid_size_full, block_size_full >> > (helper_cuda);
    }

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	cudaMemcpy(plane_hypotheses_host.get(), plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(weak_info_host.ptr<uchar>(0), weak_info_cuda, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
    if (problem.params.geom_consistency || problem.params.use_APD) {
        cudaMemcpy(confidence_host.ptr<uchar>(0), confidence_cuda, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
    }
}