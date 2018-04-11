#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/random.h>
#include <thrust/extrema.h>

/***********************/
/* BOUNDING BOX STRUCT */
/***********************/
struct bbox
{
	float3 lower_left, upper_right;

	// --- Empty box constructor
	__host__ __device__ bbox() {}

	// --- Construct a box from a single point
	__host__ __device__ bbox(const float3 &point) : lower_left(point), upper_right(point) {}

	// --- Construct a box from a pair of points
	__host__ __device__	bbox(const float3 &ll, const float3 &ur) : lower_left(ll), upper_right(ur) {}

};

/*********************************/
/* BOUNDING BOX REDUCTION STRUCT */
/*********************************/
// --- Reduce a pair of bounding boxes (a, b) to a bounding box containing a and b
struct bbox_reduction : public thrust::binary_function<bbox, bbox, bbox>
{
	__host__ __device__ bbox operator()(bbox a, bbox b)
	{
		// --- Lower left corner
		float3 ll = make_float3(thrust::min(a.lower_left.x, b.lower_left.x), thrust::min(a.lower_left.y, b.lower_left.y), thrust::min(a.lower_left.z, b.lower_left.z));

		// --- Upper right corner
		float3 ur = make_float3(thrust::max(a.upper_right.x, b.upper_right.x), thrust::max(a.upper_right.y, b.upper_right.y), thrust::max(a.upper_right.z, b.upper_right.z));

		return bbox(ll, ur);
	}
};

/********/
/* MAIN */
/********/
int main(void)
{
	const size_t N = 40;
	thrust::default_random_engine rng;
	thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

	// --- Allocate space for 3D points
	thrust::device_vector<float3> d_points(N);

	// --- Generate random 3D points in the unit cube
	for (size_t i = 0; i < N; i++)
	{
		float x = u01(rng);
		float y = u01(rng);
		float z = u01(rng);
		d_points[i] = make_float3(x, y, z);
	}

	// --- The initial bounding box contains the first point of the point cloud
	bbox init = bbox(d_points[0], d_points[0]);

	// --- Binary reduction operation
	bbox_reduction binary_op;

	// --- Compute the bounding box for the point set
	bbox result = thrust::reduce(d_points.begin(), d_points.end(), init, binary_op);

	for (int k = 0; k < N; k++) {
		float3 temp = d_points[k];
		printf("%d %f %f %f\n", k, temp.x, temp.y, temp.z);
	}
	
	// --- Print output
	std::cout << "bounding box " << std::fixed;
	std::cout << "(" << result.lower_left.x << "," << result.lower_left.y << "," << result.lower_left.z << ") ";
	std::cout << "(" << result.upper_right.x << "," << result.upper_right.y << "," << result.upper_right.z << ")" << std::endl;

	return 0;
}
