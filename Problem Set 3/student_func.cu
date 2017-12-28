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

template<typename device_reduce_operator>
__global__ void cuda_reduce(const float * vec, const size_t length, float * out, device_reduce_operator op)
{
	extern __shared__ float s_vect[];
	const int tid = threadIdx.x;
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Copy into shared memory
	s_vect[tid] = vec[position];
	__syncthreads();	

	for(int s = blockDim.x / 2; s > 0; s /= 2)
	{
		if(tid < s)
			s_vect[tid] = op(s_vect[tid], s_vect[tid + s]);
		__syncthreads();	
	}

	if(tid == 0)
	{	const float data = s_vect[0];
		out[blockIdx.x] = data;
	}
}

template<typename device_reduce_operator>
__global__ void cuda_reduce_monothread(float * vec, const size_t length, device_reduce_operator op)
{	
	float to_ret = vec[0];
	for(int i = 1; i < length; ++i)
		to_ret = min(vec[i], to_ret);
	vec[0] = to_ret;
}

template<typename device_reduce_operator>
float reduce(const float * d_vec, int length, int num_threads_per_block, device_reduce_operator op)
{	
	int num_blocks = static_cast<int>(ceil(length / (double)num_threads_per_block));
	if(num_blocks == 0) num_blocks = 1;


	float * d_out = NULL;
	checkCudaErrors(cudaMalloc(&d_out, sizeof(float) * num_blocks));
	cuda_reduce<<<num_blocks, num_threads_per_block, sizeof(float)*num_threads_per_block>>>(d_vec, length, d_out, op);
	checkCudaErrors(cudaGetLastError());		
	
	//using CUDA so that the device operator can be always used.
	cuda_reduce_monothread<<<1,1>>>(d_out, num_blocks, op);

	checkCudaErrors(cudaGetLastError());
	
	float h_out; 
	checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_out));
	
	return h_out;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
   const int K = 32;

   min_logLum = reduce(d_logLuminance, numRows * numCols, K, []__device__(float a, float b){return min(a,b);}); 
   max_logLum = reduce(d_logLuminance, numRows * numCols, K, []__device__(float a, float b){return max(a,b);}); 
   printf("%f %f", min_logLum, max_logLum);
}
