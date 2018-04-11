#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/random.h>
#include <thrust/extrema.h>

#include "octcode.h"

#include "TimingGPU.cuh"

typedef long long pointCodeType;

#define MAX_LEVEL 20

#define BLOCKSIZE 256

/*******************/
/* iDivUp FUNCTION */
/*******************/
__host__ __device__ int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}

void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/***********************/
/* BOUNDING BOX STRUCT */
/***********************/
struct bbox
{
	float3 lowerLeft, upperRight;

	// --- Empty box constructor
	__host__ __device__ bbox() {}

	// --- Construct a box from a single point
	__host__ __device__ bbox(const float3 &point) : lowerLeft(point), upperRight(point) {}

	// --- Construct a box from a pair of points
	__host__ __device__	bbox(const float3 &ll, const float3 &ur) : lowerLeft(ll), upperRight(ur) {}

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
		float3 ll = make_float3(thrust::min(a.lowerLeft.x, b.lowerLeft.x), thrust::min(a.lowerLeft.y, b.lowerLeft.y), thrust::min(a.lowerLeft.z, b.lowerLeft.z));

		// --- Upper right corner
		float3 ur = make_float3(thrust::max(a.upperRight.x, b.upperRight.x), thrust::max(a.upperRight.y, b.upperRight.y), thrust::max(a.upperRight.z, b.upperRight.z));

		return bbox(ll, ur);
	}
};

/**********************************/
/* MORTON ENCODER KERNEL FUNCTION */
/**********************************/
__global__ void mortonEncoder(const float3 * __restrict__ d_points, pointCodeType * __restrict__ d_mortonCode, const bbox boundingBox, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	/*
	000   000
	100   001
	010   010
	110   011
	001   100
	101   101
	011   110
	111   111
	*/
	
	float3 lowerLeft  = boundingBox.lowerLeft;
	float3 upperRight = boundingBox.upperRight;

	float3 d_currentPoint = d_points[tid];
	
	pointCodeType allNode = 0;
	
	for (int i = MAX_LEVEL; i >= 0; --i) {
		allNode <<= 3;
		if (d_currentPoint.x + d_currentPoint.x > lowerLeft.x + upperRight.x) {
			allNode |= 1;
			lowerLeft.x = (lowerLeft.x + upperRight.x) * 0.5f;
		}
		else
			upperRight.x = (lowerLeft.x + upperRight.x) * 0.5f;
		
		if (d_currentPoint.y + d_currentPoint.y > lowerLeft.y + upperRight.y) {
			allNode |= 2;
			lowerLeft.y = (lowerLeft.y + upperRight.y) * 0.5f;
		}
		else
			upperRight.y = (lowerLeft.y + upperRight.y) * 0.5f;
		
		if (d_currentPoint.z + d_currentPoint.z > lowerLeft.z + upperRight.z) {
			allNode |= 4;
			lowerLeft.z = (lowerLeft.z + upperRight.z) * 0.5f;
		}
		else
			upperRight.z = (lowerLeft.z + upperRight.z) * 0.5f;
	}
	
	d_mortonCode[tid] = allNode;

}

/********/
/* MAIN */
/********/
int main(void)
{
	TimingGPU timerGPU;
	
	const size_t N = 3000000;
	thrust::default_random_engine rng;
	thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

	// --- Allocate space for 3D points
	thrust::device_vector<float3> d_points(N);

	// --- Allocate space for the Morton codes on the host and on the device
	//pointCodeType *h_mortonCode		 = (pointCodeType *)malloc(N * sizeof(pointCodeType));
	//pointCodeType *h_mortonCodeCheck = (pointCodeType *)malloc(N * sizeof(pointCodeType));
	pointCodeType *d_mortonCode; gpuErrchk(cudaMalloc(&d_mortonCode, N * sizeof(pointCodeType)));

	// --- Generate random 3D points in the unit cube
	for (size_t i = 0; i < N; i++)
	{
		float x = u01(rng);
		float y = u01(rng);
		float z = u01(rng);
		d_points[i] = make_float3(x, y, z);
		printf("%d\n", i);
	}

	// --- Move the points from device to host
	thrust::host_vector<float3> h_points = d_points;

	// --- The initial bounding box contains the first point of the point cloud
	bbox init = bbox(d_points[0], d_points[0]);

	// --- Binary reduction operation
	bbox_reduction binary_op;

	// --- Compute the bounding box on the device for the point set
	timerGPU.StartCounter();
	bbox boundingBox = thrust::reduce(d_points.begin(), d_points.end(), init, binary_op);
	std::cout << "Timing calculation of the bounding box = " << timerGPU.GetCounter() << " [ms]\n";

	// --- Compute the bounding box on the host for the point set
	//float3 lowerLeftCheck  = d_points[0];
	//float3 upperRightCheck = d_points[0];
	//for (int k = 1; k < N; k++) {
	//	lowerLeftCheck.x = min(lowerLeftCheck.x, h_points[k].x);
	//	lowerLeftCheck.y = min(lowerLeftCheck.y, h_points[k].y);
	//	lowerLeftCheck.z = min(lowerLeftCheck.z, h_points[k].z);

	//	upperRightCheck.x = max(upperRightCheck.x, h_points[k].x);
	//	upperRightCheck.y = max(upperRightCheck.y, h_points[k].y);
	//	upperRightCheck.z = max(upperRightCheck.z, h_points[k].z);
	//}

	// --- Compute Morton codes
	timerGPU.StartCounter();
	mortonEncoder<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(thrust::raw_pointer_cast(d_points.data()), d_mortonCode, boundingBox, N);
	std::cout << "Timing calculation of the Morton code = " << timerGPU.GetCounter() << " [ms]\n";
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// --- Move the Morton codes from device to host
	//gpuErrchk(cudaMemcpy(h_mortonCode, d_mortonCode, N * sizeof(pointCodeType), cudaMemcpyDeviceToHost));

	/***********************************************/
	/* PRELAGACY HOST SIDE MORTON CODE COMPUTATION */
	/***********************************************/
//	float *h_points_ptr = (float *)thrust::raw_pointer_cast(h_points.data());
//	float bbMin[3] = { 1e36, 1e36, 1e36 };
//	float bbMax[3] = { -1e36,-1e36,-1e36 };
//	for (int k = 0; k < N; k++) {
//		for (int j = 0; j < 3; j++) {
//			if (h_points_ptr[k * 3 + j] < bbMin[j]) bbMin[j] = h_points_ptr[k * 3 + j];
//			if (h_points_ptr[k * 3 + j] > bbMax[j]) bbMax[j] = h_points_ptr[k * 3 + j];
//		}
//	}
//	OctCode oc(bbMin, bbMax);
//	for (int k = 0; k < N; k++) {
//#if USE_HILBERT
//		vCode[k] = oc.hilbert(vertices + k * 3);
//#else    
//		h_mortonCodeCheck[k] = oc.zOrder(h_points_ptr + k * 3);
//#endif
//	}
//	
//	printf("Points with Morton codes\n");
//	for (int k = 0; k < N; k++) {
//		float3 temp = d_points[k];
//		std::cout << k << " " << h_mortonCode[k] << " " << h_mortonCodeCheck[k] << " " << temp.x << " " << temp.y << " " << temp.z << "\n";
//		//printf("%d %f %f %f\n", k, temp.x, temp.y, temp.z);
//	}
//
//	// --- Print output
//	std::cout << "Bounding box computed on the device               " << std::fixed;
//	std::cout << "(" << boundingBox.lowerLeft.x  << "," << boundingBox.lowerLeft.y  << "," << boundingBox.lowerLeft.z << ") ";
//	std::cout << "(" << boundingBox.upperRight.x << "," << boundingBox.upperRight.y << "," << boundingBox.upperRight.z << ")" << std::endl;
//
//	std::cout << "Bounding box computed on the host                 " << std::fixed;
//	std::cout << "(" << lowerLeftCheck.x << "," << lowerLeftCheck.y << "," << lowerLeftCheck.z << ") ";
//	std::cout << "(" << upperRightCheck.x << "," << upperRightCheck.y << "," << upperRightCheck.z << ")" << std::endl;
//
//	std::cout << "Pre-legacy bounding box computation on the host   " << std::fixed;
//	std::cout << "(" << bbMin[0] << "," << bbMin[1] << "," << bbMin[2] << ") ";
//	std::cout << "(" << bbMax[0] << "," << bbMax[1] << "," << bbMax[2] << ")" << std::endl;

	return 0;
}
