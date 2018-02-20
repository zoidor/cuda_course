//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <map>
#include <algorithm>
#include <type_traits>
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

/********************Forward Declarations ***********************/

template<typename T, typename device_host_reduce_op>
__global__ static void cuda_reduce(const T * vec, const size_t length, const size_t tile_dim, T * out, device_host_reduce_op op);

template<typename T, typename device_host_reduce_op>
static T  reduce(const T * d_vec, int length, int num_threads_per_block, device_host_reduce_op op);

template<typename T, typename device_scan_operator>
__global__ static void cuda_scan_in_block(const T * d_in, T * d_out, T * d_out_tails, const size_t length_vec, device_scan_operator op, T identity);

template<typename T, typename device_scan_operator>
__global__ static void cuda_scan_post_process(const T * in_vec, const T * in_vec_tails, T * out_vec, const size_t length_vec, device_scan_operator op, unsigned int cuda_scan_in_block_b_size);

template<typename T, typename operatorType>
static void scan(T * d_vec, const size_t length, T identity_element, operatorType op);

template<typename Tv, typename T, typename OpType>
__global__ static void cuda_map2(const Tv * vals, T * out_true, T * out_false, size_t numElems, OpType op);

template<typename Tv, typename T>
__global__ static void scatter2(const T * scatter_0, const T * scatter_1, const T * flags_0, const Tv * in1, Tv * out1, const Tv * in2, Tv * out2, const size_t length);

template<typename T>
__global__ void gen_scatter1(const T * scatter0, T * scatter1, const T * flags, const size_t length);

/* Public function */

static void sort_thrust(unsigned int* const d_inputVals, 
			unsigned int* const d_inputPos, 
			unsigned int* const d_outputVals, 
			unsigned int* const d_outputPos, 
			size_t numElems)
{
		checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

//Implementation from: https://code.google.com/archive/p/back40computing/wikis/RadixSorting.wiki
//alternatively, one can take a look at the scan http://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
	thrust::device_ptr<unsigned int> p_Pos(d_outputPos);
	thrust::device_ptr<unsigned int> p_Vals(d_outputVals);

	thrust::sort_by_key(thrust::cuda::par, p_Vals, p_Vals + numElems, p_Pos); 
}

/**
 * @brief Entry point function, provided by the HW. 
 *
 * This functions takes a two arrays and sort them in the two output arrays according to the values of the first input array.
 * 
 * @param d_inputVals:  input, values the sorting is going to be done on
 * @param d_inputPos:   input, positions to be sorted according to the corresponding element in d_inputVals
 * @param d_outputVals: output, sorted d_inputVals
 * @param d_outputPos:  output, sorted d_outputPos according to the corresponding element in d_inputVals
 * @param numElems:    	input, number of elements
 */
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               size_t numElems,
	       const bool use_thrust)
{ 	

	if(use_thrust){
		sort_thrust(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
		return;
	}
	
	using type_scatter = int;

	type_scatter * scatter_loc0 = NULL;
	type_scatter * scatter_loc1 = NULL;
	type_scatter * flags = NULL;

	checkCudaErrors(cudaMalloc(&scatter_loc0, sizeof(type_scatter) * (1 + numElems)));
	checkCudaErrors(cudaMalloc(&flags, sizeof(type_scatter) * numElems));
	checkCudaErrors(cudaMalloc(&scatter_loc1, sizeof(type_scatter) * (1 + numElems)));

	const int K = 1024;
	const int K_reduce = 512;
	const int num_blocks = (int)ceil(numElems / (double)K);

	auto scan_op = [] __device__ (type_scatter el1, type_scatter el2) -> type_scatter {return el1 + el2;};


	auto max_op = []__device__ __host__ (type_scatter el1, type_scatter el2) -> type_scatter {return max(el1, el2);};
	
	const int max_el = reduce(d_inputVals, numElems, K_reduce, max_op);
	const int numbits = std::ceil(log2((double)max_el));

	unsigned int * vals1 = d_inputVals;
	unsigned int * pos1 = d_inputPos;

	unsigned int * vals2 = d_outputVals;
	unsigned int * pos2 = d_outputPos;

	for(int i = 0; i < numbits; ++i)
	{	unsigned int mask = pow(2, i);
		auto mask_op_0 = [mask]__device__ (unsigned int el) -> type_scatter
			  {
				return (type_scatter)((el & mask) == 0);
			  };
		
		cuda_map2<<<num_blocks, K>>>(vals1, scatter_loc0, scatter_loc1, numElems, mask_op_0);
		checkCudaErrors(cudaGetLastError());		
		
		checkCudaErrors(cudaMemcpy(flags, scatter_loc0, sizeof(type_scatter) * numElems, cudaMemcpyDeviceToDevice));
		
		scan(scatter_loc0, numElems + 1, (type_scatter)0, scan_op);
		checkCudaErrors(cudaGetLastError());		

		gen_scatter1<<<num_blocks, K>>>(scatter_loc0, scatter_loc1, flags, numElems);
		checkCudaErrors(cudaGetLastError());		
		
		scatter2<<<num_blocks, K>>>(scatter_loc0, scatter_loc1, flags, vals1, vals2, pos1, pos2, numElems);

		std::swap(vals1, vals2);
		std::swap(pos1, pos2);
	}

	checkCudaErrors(cudaFree(scatter_loc0));
	checkCudaErrors(cudaFree(scatter_loc1));
	checkCudaErrors(cudaFree(flags));

	checkCudaErrors(cudaMemcpy(d_outputVals, vals1, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, pos1, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

}


/*Private functions implementation*/

template<typename T, typename device_host_reduce_op>
static T  reduce(const T * d_vec, int length, int num_threads_per_block, device_host_reduce_op op)
{
	const size_t tile_size = 2;	
	int num_blocks = static_cast<int>(ceil(length / (double)num_threads_per_block / (double)tile_size / 2.0));
	if(num_blocks == 0) num_blocks = 1;

	T * d_out = NULL;
	checkCudaErrors(cudaMalloc(&d_out, sizeof(T) * num_blocks));

	cuda_reduce<<<num_blocks, num_threads_per_block, sizeof(T) * num_threads_per_block>>>(d_vec, length, tile_size, d_out, op);
	checkCudaErrors(cudaGetLastError());		
	
	std::vector<T> h_out(num_blocks); 
	checkCudaErrors(cudaMemcpy(h_out.data(), d_out, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_out));
	
	return std::accumulate(h_out.cbegin() + 1, h_out.cend(), h_out[0], op);
}


template<typename T, typename device_host_reduce_op>
__device__ T reduce_tile(const T * vec, const size_t length, const size_t tile_dim, device_host_reduce_op op, const size_t start)
{
	const size_t stop = min(length - 1, start + tile_dim - 1);
	
	T e = vec[start];
	for(size_t i = start + 1; i <= stop; ++i){
		e = op(e, vec[i]);
	}
	return e;
}

template<typename T, typename device_host_reduce_op>
__global__ static void cuda_reduce(const T * vec, const size_t length, const size_t tile_dim, T * out, device_host_reduce_op op)
{
	extern __shared__ T s_vect[];

	const int tid = threadIdx.x;
	const size_t start1 = (threadIdx.x + blockIdx.x * blockDim.x) * tile_dim * 2;
	if(start1 > length) return;

	//Copy into shared memory
	s_vect[tid] = reduce_tile(vec, length, tile_dim, op, start1);

	const size_t start2 = (threadIdx.x + blockIdx.x * blockDim.x) * tile_dim * 2 + tile_dim;
	if(start2 <= length - 1)
		s_vect[tid] = op(s_vect[tid], reduce_tile(vec, length, tile_dim, op, start2));

	__syncthreads();	

	for(int s = blockDim.x / 2; s > 0; s /= 2)
	{
		if(tid < s)
		{
			s_vect[tid] = op(s_vect[tid], s_vect[tid + s]);
		}
		__syncthreads();	
	}

	if(tid == 0)
	{	
		atomicAdd(out, s_vect[0]);
	}
}


template<typename T>
class CudaBuffer
{
public:

	CudaBuffer() : CudaBuffer(0){}

	CudaBuffer(const size_t num_els) : m_num_els(num_els)
	{
		if(num_els != 0)
			checkCudaErrors(cudaMalloc(&m_d_buffer, sizeof(T) * num_els)); 	
	}
	

	~CudaBuffer()
	{	
		if(m_d_buffer == NULL || m_num_els == 0) return;

		checkCudaErrors(cudaFree(m_d_buffer));
		m_d_buffer = NULL;
		m_num_els = 0;
	}
	

	T* getDeviceBuffer()const {return m_d_buffer;}
	size_t getNumEls()const {return m_num_els;}

	CudaBuffer(const CudaBuffer<T>&) = delete; 
	CudaBuffer<T>& operator=(const CudaBuffer<T>&) = delete; 

	CudaBuffer(CudaBuffer<T>&& other)
	{
		std::swap(other.m_d_buffer, m_d_buffer);
		std::swap(other.m_num_els, m_num_els);
	}

	CudaBuffer<T>& operator=(CudaBuffer<T>&& other)
	{	
		if(this == &other) return *this;

		this->~CudaBuffer<T>();
		std::swap(other.m_d_buffer, m_d_buffer);
		std::swap(other.m_num_els, m_num_els);
		return *this;
	}

private:
	T* m_d_buffer = NULL;
	size_t m_num_els = 0;
};

template<typename T>
class CudaBufferPool
{
	public:
		CudaBufferPool(const size_t num_elems_max) : m_max_els(num_elems_max){}
		
		void add(CudaBuffer<T> other)
		{
			if(other.getNumEls() == 0) return;

			if(buffers.size() >= m_max_els) return;
						
			buffers.emplace_back(std::move(other));
		}

		CudaBuffer<T> get(const size_t sz)
		{
			auto it = std::find_if(buffers.begin(), buffers.end(), [sz](const CudaBuffer<T>& el){return el.getNumEls() == sz;});
			if(it == buffers.end())
				return CudaBuffer<T>(sz);

			auto to_ret = std::move(*it);
			buffers.erase(it);
			return to_ret;
		}

	private:
		std::vector<CudaBuffer<T>> buffers;
		size_t m_max_els;
};



template<typename T, typename device_scan_operator>
__global__ static void cuda_scan_in_block(const T * d_in, T * d_out, T * d_out_tails, const size_t length_vec, device_scan_operator op, T identity)
{
	extern __shared__ T s_block_scan_mem[];
	T * s_block_scan1 = s_block_scan_mem;
	const int tid = threadIdx.x;
	const int pos = tid + blockDim.x * blockIdx.x;

	if(pos >= length_vec) return;

	s_block_scan1[tid] = pos == 0 ? identity : d_in[pos -1];
	__syncthreads();

	for(int shift = 1; shift < blockDim.x; shift *= 2)
	{
		const int prev = tid - shift;
		T el = s_block_scan1[tid];
		if(prev >= 0)
			el = op(el, s_block_scan1[prev]);
		__syncthreads();
		 s_block_scan1[tid] = el;
		__syncthreads();
	} 

	d_out[pos] = s_block_scan1[tid];

	if(tid == blockDim.x - 1) 
	{
		d_out_tails[blockIdx.x] =  s_block_scan1[tid];
	}
}

template<typename T, typename device_scan_operator>
__global__ static  void cuda_scan_post_process(const T * in_vec, const T * in_vec_tails, T * out_vec, const size_t length_vec, device_scan_operator op, unsigned int cuda_scan_in_block_b_size)
{
	const size_t pos = blockDim.x * blockIdx.x + threadIdx.x;

	if(pos >= length_vec) 
	{
		return;
	}

	//we want to allow the previous and current block sizes to be different
	const int idx_of_block_in_scan = pos / cuda_scan_in_block_b_size;
	T el = op(in_vec[pos], in_vec_tails[idx_of_block_in_scan]);
	out_vec[pos] = el;
}


template<typename T, typename operatorType>
static void scan(T * d_vec, const size_t length, T identity_element, operatorType op)
{

	static CudaBufferPool<T> pool(20);
	const int K = 128;
	const int K2 = 1024;
	const int num_blocks = std::max(1, static_cast<int>(std::ceil(length / (double)K)));
	const int num_blocks2 = std::max(1, static_cast<int>(std::ceil(length / (double)K2)));


	CudaBuffer<T> d_out_vec_tails(pool.get(num_blocks));
	CudaBuffer<T> d_out_vec(pool.get(length));
	
	cuda_scan_in_block<<<num_blocks, K, sizeof(T) * K>>>(d_vec, d_out_vec.getDeviceBuffer(), d_out_vec_tails.getDeviceBuffer(), length, op, identity_element);
	checkCudaErrors(cudaGetLastError());

	if(num_blocks != 1)
	{
		scan(d_out_vec_tails.getDeviceBuffer(), num_blocks, identity_element, op);
		cuda_scan_post_process<<<num_blocks2, K2>>>(d_out_vec.getDeviceBuffer(), d_out_vec_tails.getDeviceBuffer(), d_vec, length, op, K);
	}
	else
	{
		checkCudaErrors(cudaMemcpy(d_vec, d_out_vec.getDeviceBuffer(), sizeof(T) * length, cudaMemcpyDeviceToDevice));
	}

	checkCudaErrors(cudaGetLastError());		

	pool.add(std::move(d_out_vec_tails));
	pool.add(std::move(d_out_vec));
}



template<typename Tv, typename T, typename OpType>
__global__ static void cuda_map2(const Tv * vals, T * out_true, T * out_false, size_t numElems, OpType op)
{
	const size_t pos = blockDim.x * blockIdx.x + threadIdx.x;
	if(pos >= numElems) return;
	
	T is_bit_active = op(vals[pos]);
	out_true[pos] = is_bit_active;
	out_false[pos] = !is_bit_active;
}

template<typename T>
__global__ void gen_scatter1(const T * scatter0, T * scatter1, const T * flags, const size_t length){
	
	size_t pos = threadIdx.x + blockDim.x * blockIdx.x;

	if(pos >= length) return;

	scatter1[pos] = pos - scatter0[pos] + scatter0[length + 1];
	
}

template<typename Tv, typename T>
__global__ static void scatter2(const T * scatter_0, const T * scatter_1, const T * flags_0, const Tv * in1, Tv * out1, const Tv * in2, Tv * out2, const size_t length)
{
	const size_t pos = threadIdx.x + blockDim.x * blockIdx.x;
	if(pos >= length) return;

	T scatter_pos = flags_0[pos] ? scatter_0[pos] : scatter_1[pos];
	
	if(scatter_pos >= length) return;

	out1[scatter_pos] = in1[pos]; 
	out2[scatter_pos] = in2[pos]; 
}  


