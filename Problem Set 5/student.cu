/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

//reference: https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
__global__
void cuda_hist(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals, 
	       int num_bins)
{
 	extern __shared__ unsigned int s_hist[];

	const int tid = threadIdx.x;
	
	for(int i = tid; i < num_bins; i += blockDim.x)
		s_hist[i] = 0;
  
	__syncthreads();

	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	if(position < numVals)
	{
		const unsigned int bin = vals[position];
		atomicAdd(&s_hist[bin], 1);
	}
	__syncthreads();

	for(int i = tid; i < num_bins; i += blockDim.x)
		atomicAdd(&histo[i], s_hist[i]);
}


__global__
void cuda_hist_naive(const unsigned int* const vals, //INPUT
               	     unsigned int* const histo,      //OUPUT
                     int numVals, 
	       	     int num_bins)
{
 	const int pos = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(pos < numVals)
	{
		const unsigned int bin = vals[pos];
		atomicAdd(&histo[bin], 1);
	}
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
	const int num_threads = 32;
	checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));
	int numBlocks = (int)ceil(numElems / (double)num_threads);
	if(numBlocks == 0) numBlocks = 1;
	cuda_hist_naive<<<numBlocks, num_threads, sizeof(unsigned int) * numBins>>> (d_vals, d_histo, 
										numElems, numBins); 

   	checkCudaErrors(cudaGetLastError());		
   	checkCudaErrors(cudaGetLastError());
}
