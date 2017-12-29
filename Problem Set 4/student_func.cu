//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


template<typename device_scan_operator>
__global__ void cuda_scan_in_block(const size_t * d_in, size_t * d_out, const size_t length_vec, device_scan_operator op, unsigned int identity)
{
	extern __shared__ size_t s_block_scan_mem[];
	size_t * s_block_scan1 = s_block_scan_mem;
	size_t * s_block_scan2 = s_block_scan_mem + blockDim.x;
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
		size_t * tmp = s_block_scan1;
		s_block_scan1 = s_block_scan2;
		s_block_scan2 = tmp;
	} 

	d_out[pos] = s_block_scan1[tid];
}

template<typename device_scan_operator>
__global__ void cuda_scan_post_process(const size_t * in_vec, size_t * out_vec, const size_t length_vec, device_scan_operator op, unsigned int cuda_scan_in_block_b_size)
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

template<typename operatorType>
void scan(size_t * const d_vec, const size_t length, unsigned int identity_element, operatorType op){
	int K = 32;
	int K2 = 64;
	int num_blocks = std::ceil(length / (double)K);
	int num_blocks2 = std::ceil(length / (double)K2);
	size_t * d_out_vec = NULL;

	checkCudaErrors(cudaMalloc(&d_out_vec, sizeof(size_t) * length));

	cuda_scan_in_block<<<num_blocks, K, sizeof(size_t) * K * 2>>>(d_vec, d_out_vec, length, op, identity_element);
	checkCudaErrors(cudaGetLastError());		
	
	cuda_scan_post_process<<<num_blocks2, K2>>>(d_out_vec, d_vec, length, op, K);
	checkCudaErrors(cudaGetLastError());		

	checkCudaErrors(cudaFree(d_out_vec));	
}


template<typename OpType>
__global__ void cuda_get_flags(const unsigned int * vals, size_t * flags, size_t numElems, OpType op)
{
	const size_t pos = blockDim.x * blockIdx.x + threadIdx.x;
	if(pos >= numElems) return;
	
	unsigned int is_bit_active = op(vals[pos]);
	flags[pos] = is_bit_active;
}


__global__ void scatter(const size_t * scatter_0, const size_t * scatter_1, const size_t * flags_0, const unsigned int * in, unsigned int * out, const size_t length)
{
	const size_t pos = threadIdx.x + blockDim.x * blockIdx.x;
	if(pos >= length) return;

	size_t scatter_pos = flags_0[pos] ? scatter_0[pos] : scatter_1[pos];
	out[scatter_pos] = in[pos]; 
}  

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
 	int numbits = sizeof(unsigned int) * 8;
	
	unsigned int * vals1 = d_inputVals;
	unsigned int * vals2 = d_outputVals;


	unsigned int * pos1 = d_inputPos;
	unsigned int * pos2 = d_outputPos;

	size_t * scatter_loc0 = NULL;
	size_t * scatter_loc1 = NULL;
	size_t * flags = NULL;

	checkCudaErrors(cudaMalloc(&scatter_loc0, sizeof(size_t) * numElems));
	checkCudaErrors(cudaMalloc(&flags, sizeof(size_t) * numElems));
	checkCudaErrors(cudaMalloc(&scatter_loc1, sizeof(size_t) * numElems));
	
	const int K = 32;
	const int num_blocks = (int)ceil(numElems / (double)K);

	auto scan_op = [] __device__ (size_t el1, size_t el2) -> size_t {return el1 + el2;};

	for(int i = 0; i < numbits; ++i)
	{	unsigned int mask = pow(2, i);
		auto mask_op_0 = [mask]__device__ (unsigned int el) -> unsigned int
			  {
				return (int)((el & mask) == 0);
			  };

		auto mask_op_1 = [mask]__device__ (unsigned int el) -> unsigned int
			  {
				return (int)((el & mask) == 1);
			  }; 

				
		cuda_get_flags<<<num_blocks, K>>>(vals1, scatter_loc0, numElems, mask_op_0);
		checkCudaErrors(cudaGetLastError());		
		
		checkCudaErrors(cudaMemcpy(flags, scatter_loc0, sizeof(size_t) * numElems, cudaMemcpyDeviceToDevice));
		scan(scatter_loc0, numElems, 0, scan_op);
		checkCudaErrors(cudaGetLastError());		
		
		size_t start1;
		checkCudaErrors(cudaMemcpy(&start1, &scatter_loc0[numElems - 1], sizeof(size_t), cudaMemcpyDeviceToHost));

		cuda_get_flags<<<num_blocks, K>>>(d_inputVals, scatter_loc1, numElems, mask_op_1);
		scan(scatter_loc1, numElems, start1, scan_op);
		checkCudaErrors(cudaGetLastError());		
				
		scatter<<<num_blocks, K>>>(scatter_loc0, scatter_loc1, flags, vals1, vals2, numElems);
		scatter<<<num_blocks, K>>>(scatter_loc0, scatter_loc1, flags, pos1, pos2, numElems);

		std::swap(vals1, vals2);
		std::swap(pos1, pos2);
	}

	checkCudaErrors(cudaFree(scatter_loc0));
	checkCudaErrors(cudaFree(scatter_loc1));
	checkCudaErrors(cudaFree(flags));

	if(vals1 == d_outputVals) return;

	checkCudaErrors(cudaMemcpy(d_outputVals, vals1, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, pos1, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
}
