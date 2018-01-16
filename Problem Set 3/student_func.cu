/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <vector>
#include <numeric>


/*Forward declaration of private functions*/
template<typename operatorType>
static void scan(unsigned int * const d_vec, const size_t length, unsigned int identity_element, operatorType op);

void generate_histogram(const float* const d_vec,
			const size_t length_vec,
			unsigned int* const d_hist,
			const size_t num_bins,
			const float vec_min,
			const float vec_max,
			int num_threads);

template<typename device_hist_operator>
__global__ static void cuda_hist(const float * d_vec, const size_t length_vec, unsigned int * d_hist, 
const size_t num_bins, device_hist_operator op);

template<typename device_scan_operator>
__global__ static void cuda_scan_in_block(const unsigned int * d_in, unsigned int * d_out, const size_t length_vec, device_scan_operator op, unsigned int identity);

template<typename device_scan_operator>
__global__ static void cuda_scan_post_process(const unsigned int * in_vec, unsigned int * out_vec, const size_t length_vec, device_scan_operator op, unsigned int cuda_scan_in_block_b_size);

template<typename T, typename device_host_reduce_op1, typename device_host_reduce_op2>
static std::tuple<T, T> reduce(const T * d_vec, int length, int num_threads_per_block, device_host_reduce_op1 op1, device_host_reduce_op2 op2);

template<typename T, typename device_host_reduce_op1, typename device_host_reduce_op2>
__global__ static void cuda_reduce(const T * vec, const size_t length, T * out1, T * out2,  device_host_reduce_op1 op1, device_host_reduce_op2 op2);

/**
 * @brief Entry point function, provided by the HW. 
 *
 * This functions takes a log-luminance image as a buffer, and calculates its cdf.
 * 
 * @param d_logLuminance:    input, image buffer in device memory. Provided by the caller code
 * @param d_cdf:             output, pre-allocated buffer for the cdf (cumulative distribution)
 * @param min_logLum:        output, minimum element in d_logLuminance
 * @param max_logLum:        output, maximum element in d_logLuminance
 * @param numRows:     	     input, number of rows of the image
 * @param numCols:           input, number of columns of the image 
 * @param numBins:           input, number of bins in the cdf 
 */
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
   const int K = 128;

   auto min_max = reduce(d_logLuminance, numRows * numCols, K, []__host__ __device__(float a, float b){return min(a,b);}, []__host__ __device__(float a, float b){return max(a,b);}); 
   min_logLum = std::get<0>(min_max);   
   max_logLum = std::get<1>(min_max);

  const int K_hist = 32;
  generate_histogram(d_logLuminance, numRows * numCols, d_cdf, numBins, min_logLum, max_logLum, K_hist);

  scan(d_cdf, numBins, 0, [] __device__(unsigned int a, unsigned int b){return a + b;});
}

/*Implementation of private functions*/


template<typename T, typename device_host_reduce_op1, typename device_host_reduce_op2>
static std::tuple<T, T> reduce(const T * d_vec, int length, int num_threads_per_block, device_host_reduce_op1 op1, device_host_reduce_op2 op2)
{	
	int num_blocks = static_cast<int>(ceil(length / (double)num_threads_per_block));
	if(num_blocks == 0) num_blocks = 1;


	T * d_out1 = NULL;
	T * d_out2 = NULL;
	checkCudaErrors(cudaMalloc(&d_out1, sizeof(T) * num_blocks));
	checkCudaErrors(cudaMalloc(&d_out2, sizeof(T) * num_blocks));

	cuda_reduce<<<num_blocks, num_threads_per_block, sizeof(T) * num_threads_per_block * 2>>>(d_vec, length, d_out1, d_out2, op1, op2);
	checkCudaErrors(cudaGetLastError());		

	
	std::vector<T> h_out1(num_blocks); 
	std::vector<T> h_out2(num_blocks); 
	checkCudaErrors(cudaMemcpy(h_out1.data(), d_out1, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_out2.data(), d_out2, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_out1));
	checkCudaErrors(cudaFree(d_out2));
	
	return std::tuple<T, T>{std::accumulate(h_out1.cbegin() + 1, h_out1.cend(), h_out1[0], op1),
		std::accumulate(h_out2.cbegin() + 1, h_out2.cend(), h_out2[0], op2)};
}


template<typename T, typename device_host_reduce_op1, typename device_host_reduce_op2>
__global__ static void cuda_reduce(const T * vec, const size_t length, T * out1, T * out2,  device_host_reduce_op1 op1, device_host_reduce_op2 op2)
{
	extern __shared__ T s_vect[];
	T * s_vect1  = s_vect;
	T * s_vect2 = s_vect + blockDim.x;

	const int tid = threadIdx.x;
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(position > length) return;

	//Copy into shared memory
	s_vect1[tid] = vec[position];
	s_vect2[tid] = vec[position];
	__syncthreads();	

	for(int s = blockDim.x / 2; s > 0; s /= 2)
	{
		if(tid < s)
		{
			s_vect1[tid] = op1(s_vect1[tid], s_vect1[tid + s]);
			s_vect2[tid] = op2(s_vect2[tid], s_vect2[tid + s]);
		}
		__syncthreads();	
	}

	if(tid == 0)
	{	
		out1[blockIdx.x] = s_vect1[0];
		out2[blockIdx.x] = s_vect2[0];
	}
}



void generate_histogram(const float* const d_vec,
			const size_t length_vec,
			unsigned int* const d_hist,
			const size_t num_bins,
			const float vec_min,
			const float vec_max,
			int num_threads)
{

	checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) * num_bins));
	int numBlocks = (int)ceil(length_vec / (double)num_threads);
	if(numBlocks == 0) numBlocks = 1;
	const float range = vec_max - vec_min;
	auto op = [num_bins, range, vec_min] __device__(float el) -> int
	{
		if(range ==  0.0f) return 0;
		return min((size_t)((el - vec_min) / range * num_bins), num_bins - 1);
	};
	cuda_hist<<<numBlocks, num_threads>>> (d_vec, length_vec, 
										d_hist, num_bins, op); 

	checkCudaErrors(cudaGetLastError());		
}

//We tried to use: https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/ ,
//i.e. we tried to create a shared histogram in shared memory and then perform a reduction step to populate the global memory.
//This simpler version that uses global memory performs better in this case. We did some performance tuning and the
//bottleneck was in the access to shared memory, not in the reduction step that we chose.
template<typename device_hist_operator>
__global__ static void cuda_hist(const float * d_vec, const size_t length_vec, unsigned int * d_hist, 
const size_t num_bins, device_hist_operator op){
	
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	if(position < length_vec)
	{
		const int bin = op(d_vec[position]);
		atomicAdd(&d_hist[bin], 1);
	}
}

/*Idea: perform scan on every block (segment), keeping track of the tails of each segment. Then scan the array of the tails. 
  For every block, add the scanned tail of the previous block. See  https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html .
  If the array of tails is too long to be scanned, do a recursion until a tails segment fits in a block. */
template<typename operatorType>
static void scan(unsigned int * const d_vec, const size_t length, unsigned int identity_element, operatorType op){
	int K = 64;
	int K2 = 128;
	int num_blocks = std::ceil(length / (double)K);
	int num_blocks2 = std::ceil(length / (double)K2);
	unsigned int * d_out_vec = NULL;

	checkCudaErrors(cudaMalloc(&d_out_vec, sizeof(unsigned int) * length));

	cuda_scan_in_block<<<num_blocks, K, sizeof(unsigned int) * K * 2>>>(d_vec, d_out_vec, length, op, identity_element);
	checkCudaErrors(cudaGetLastError());		
	
	cuda_scan_post_process<<<num_blocks2, K2>>>(d_out_vec, d_vec, length, op, K);
	checkCudaErrors(cudaGetLastError());		

	checkCudaErrors(cudaFree(d_out_vec));	
}

//reference: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
//We assume that the number of bins is << than the number of processor. Hence we use a step efficient scan (Hillis & Steel). 
//In a real scenario, we should implement both, check on various workloads and architectures. Maybe come up with an heuristic 
//to switch between the two implementations at compile-time
template<typename device_scan_operator>
__global__ static void cuda_scan_in_block(const unsigned int * d_in, unsigned int * d_out, const size_t length_vec, device_scan_operator op, unsigned int identity)
{
	extern __shared__ unsigned int s_block_scan_mem[];
	unsigned int * s_block_scan1 = s_block_scan_mem;
	unsigned int * s_block_scan2 = s_block_scan_mem + blockDim.x;
	const int tid = threadIdx.x;
	const int pos = tid + blockDim.x * blockIdx.x;

	if(pos >= length_vec) return;

	s_block_scan1[tid] = pos == 0 ? identity : d_in[pos -1];
	__syncthreads();

	for(int shift = 1; shift <= blockDim.x; shift *= 2)
	{
		const int prev = tid - shift;
		if(prev >= 0)
			s_block_scan2[tid] = op(s_block_scan1[tid], s_block_scan1[prev]);
		else
			s_block_scan2[tid] = s_block_scan1[tid];
		__syncthreads();
		unsigned int * tmp = s_block_scan1;
		s_block_scan1 = s_block_scan2;
		s_block_scan2 = tmp;
	} 

	d_out[pos] = s_block_scan1[tid];
}

template<typename device_scan_operator>
__global__ static void cuda_scan_post_process(const unsigned int * in_vec, unsigned int * out_vec, const size_t length_vec, device_scan_operator op, unsigned int cuda_scan_in_block_b_size)
{
	unsigned int pos = blockDim.x * blockIdx.x + threadIdx.x;

	if(pos >= length_vec) return;

	//we want that the previous and current block sizes can be different
	const int idx_of_block_in_scan = pos / cuda_scan_in_block_b_size;
	int last_el_of_prev_block = idx_of_block_in_scan * cuda_scan_in_block_b_size - 1;

	//this is linear scan to put together the segment tails, I am assuming that the number 
	//of blocks is relatively small hence this loop is not a problem. If the number of blocks is
	//high, we generate an array of the segment tails and use scan on it. The recursion ends when
	//we can execute the scan on a single block
	out_vec[pos] = in_vec[pos];
	for(; last_el_of_prev_block > 0; last_el_of_prev_block -= cuda_scan_in_block_b_size)
	{
		out_vec[pos] += in_vec[last_el_of_prev_block];
	}
}

