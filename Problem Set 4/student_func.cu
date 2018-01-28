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

//TODO: investigate wether the kernels can be reduced. Investigate whether block size can be used to improve performance. 
template<typename device_scan_operator>
__global__ void cuda_scan_in_block(const size_t * d_in, size_t * d_out, size_t * d_out_tails, const size_t length_vec, device_scan_operator op, unsigned int identity)
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

	if(tid == blockDim.x - 1) 
	{
		d_out_tails[blockIdx.x] =  s_block_scan1[tid];
	}
}

template<typename device_scan_operator>
__global__ void cuda_scan_post_process(const size_t * in_vec, const size_t * in_vec_tails, size_t * out_vec, const size_t length_vec, device_scan_operator op, unsigned int cuda_scan_in_block_b_size, const size_t start)
{
	const size_t pos = blockDim.x * blockIdx.x + threadIdx.x;

	if(pos >= length_vec) 
	{
		return;
	}

	//we want to allow the previous and current block sizes to be different
	const int idx_of_block_in_scan = pos / cuda_scan_in_block_b_size;
	size_t el = op(in_vec[pos], in_vec_tails[idx_of_block_in_scan]);
	out_vec[pos] = op(el, start);
}

void print(const size_t * d_vec, const size_t length)
{
	std::vector<size_t> toP(length);
	cudaMemcpy(toP.data(), d_vec, sizeof(size_t) * length, cudaMemcpyDeviceToHost);
	for(auto el : toP)	
	{
		printf("%i ", (int)el);
	}
	printf("\n--------------- %i *------\n", (int)length);
}

template<typename operatorType>
void scan(size_t * const d_vec, const size_t length, unsigned int identity_element, operatorType op, unsigned int start_element){
	const int K = 1024;
	const int K2 = 1024;
	const int num_blocks = std::max(1, static_cast<int>(std::ceil(length / (double)K)));
	const int num_blocks2 = std::max(1, static_cast<int>(std::ceil(length / (double)K2)));
	size_t * d_out_vec = NULL;
	size_t * d_out_vec_tails = NULL;

	checkCudaErrors(cudaMalloc(&d_out_vec, sizeof(size_t) * length));
	checkCudaErrors(cudaMalloc(&d_out_vec_tails, sizeof(size_t) * num_blocks));
	cuda_scan_in_block<<<num_blocks, K, sizeof(size_t) * K * 2>>>(d_vec, d_out_vec, d_out_vec_tails, length, op, identity_element);
	checkCudaErrors(cudaGetLastError());

	if(num_blocks == 1)
	{

		cudaMemcpy(d_vec, d_out_vec, length * sizeof(size_t), cudaMemcpyDeviceToDevice);
	}
	else
	{
		scan(d_out_vec_tails, num_blocks, identity_element, op, identity_element);
		cuda_scan_post_process<<<num_blocks2, K2>>>(d_out_vec, d_out_vec_tails, d_vec, length, op, K, start_element);
	}

	checkCudaErrors(cudaGetLastError());		

	checkCudaErrors(cudaFree(d_out_vec));	
	checkCudaErrors(cudaFree(d_out_vec_tails));	
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
	
	if(scatter_pos >= length) return;

	out[scatter_pos] = in[pos]; 
}  


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               size_t numElems)
{ 

	//numElems = 32;
 	int numbits = sizeof(unsigned int) * 8;
	
	unsigned int * vals1 = NULL;
	unsigned int * vals2 = NULL;


	unsigned int * pos1 = NULL;
	unsigned int * pos2 = NULL;

	

	size_t * scatter_loc0 = NULL;
	size_t * scatter_loc1 = NULL;
	size_t * flags = NULL;

	checkCudaErrors(cudaMalloc(&scatter_loc0, sizeof(size_t) * numElems));
	checkCudaErrors(cudaMalloc(&flags, sizeof(size_t) * numElems));
	checkCudaErrors(cudaMalloc(&scatter_loc1, sizeof(size_t) * numElems));


	checkCudaErrors(cudaMalloc(&vals1, sizeof(unsigned int) * numElems));
	checkCudaErrors(cudaMalloc(&vals2, sizeof(unsigned int) * numElems));
	checkCudaErrors(cudaMalloc(&pos1, sizeof(unsigned int) * numElems));
	checkCudaErrors(cudaMalloc(&pos2, sizeof(unsigned int) * numElems));
	
	checkCudaErrors(cudaMemcpy(vals1, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(pos1, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

	const int K = 512;
	const int num_blocks = (int)ceil(numElems / (double)K);

	auto scan_op = [] __device__ (size_t el1, size_t el2) -> size_t {return el1 + el2;};


	for(int i = 0; i < numbits; ++i)
	{	unsigned int mask = pow(2, i);
		auto mask_op_0 = [mask]__device__ (unsigned int el) -> unsigned int
			  {
				return (unsigned int)((el & mask) == 0);
			  };

		auto mask_op_1 = [mask]__device__ (unsigned int el) -> unsigned int
			  {
				return (unsigned int)((el & mask) != 0);
			  }; 

		
		cuda_get_flags<<<num_blocks, K>>>(vals1, scatter_loc0, numElems, mask_op_0);
		checkCudaErrors(cudaGetLastError());		
		
		checkCudaErrors(cudaMemcpy(flags, scatter_loc0, sizeof(size_t) * numElems, cudaMemcpyDeviceToDevice));
		
		scan(scatter_loc0, numElems, 0, scan_op, 0);
		checkCudaErrors(cudaGetLastError());		
		

		size_t start1;
		checkCudaErrors(cudaMemcpy(&start1, &scatter_loc0[numElems - 1], sizeof(size_t), cudaMemcpyDeviceToHost));

		size_t is_last_1;		
		checkCudaErrors(cudaMemcpy(&is_last_1, &flags[numElems - 1], sizeof(size_t), cudaMemcpyDeviceToHost));
		
		start1 += is_last_1;
		cuda_get_flags<<<num_blocks, K>>>(vals1, scatter_loc1, numElems, mask_op_1);
		scan(scatter_loc1, numElems, 0, scan_op, start1);
		checkCudaErrors(cudaGetLastError());		
				
		scatter<<<num_blocks, K>>>(scatter_loc0, scatter_loc1, flags, vals1, vals2, numElems);
		scatter<<<num_blocks, K>>>(scatter_loc0, scatter_loc1, flags, pos1, pos2, numElems);

		std::swap(vals1, vals2);
		std::swap(pos1, pos2);
	}

	checkCudaErrors(cudaFree(scatter_loc0));
	checkCudaErrors(cudaFree(scatter_loc1));
	checkCudaErrors(cudaFree(flags));

	checkCudaErrors(cudaMemcpy(d_outputVals, vals1, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, pos1, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaFree(vals1));
	checkCudaErrors(cudaFree(vals2));
	checkCudaErrors(cudaFree(pos1));
	checkCudaErrors(cudaFree(pos2));
}
