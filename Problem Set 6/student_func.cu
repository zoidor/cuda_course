//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include <algorithm>

const unsigned char non_mask_region = 0;
const unsigned char mask_region = 2; //unknown if border or or border
const unsigned char mask_region_interior = 1; 
const unsigned char mask_region_border = 3;

/*----------------- Private Functions forward declaration -----------------*/

template<typename device_scan_operator>
__global__ static void cuda_scan_in_block(const size_t * d_in, size_t * d_out, size_t * d_out_tails, const size_t length_vec, device_scan_operator op, unsigned int identity);

template<typename device_scan_operator>
__global__ static  void cuda_scan_post_process(const size_t * in_vec, const size_t * in_vec_tails, size_t * out_vec, const size_t length_vec, device_scan_operator op, unsigned int cuda_scan_in_block_b_size, const size_t start);

template<typename operatorType>
static void scan(size_t * const d_vec, const size_t length, unsigned int identity_element, operatorType op, unsigned int start_element);

template<typename T1, typename T2>
void copy(T1 * dest, const T2 * source, std::size_t sz);

__global__ void cuda_compact(size_t * x_arr, size_t * y_arr, size_t * mask_scan, const size_t length_mask_compacted, const unsigned char * mask, const size_t sz_x, const size_t sz_y);

template<typename OType, typename IType, typename OperatorType>
__global__ void map(OType * out, const  IType * in, const size_t sz, OperatorType op);

class Positions;

Positions compact_interior_border(const unsigned char * d_refined_mask, const size_t numColsSource, const size_t numRowsSource);

void generate_mask(unsigned char * mask, const uchar4 * img, const size_t sz);

template<typename opType>
void extract_channel(unsigned char * out_img, const uchar4 * img, const size_t sz, opType op);

__device__ void add_neigh(const unsigned char * in_mask, const size_t x_sz, const size_t y_sz, const int x, const int y, const int dx, const int dy, int &num_valid_neigh, int &num_masked_neigh);

void calculate_interior_border(const unsigned char * in_mask, unsigned char * out_mask, const size_t x_sz, const size_t y_sz);

__device__ void perform_j_neigh(const float * b1, const unsigned char * source, const unsigned char * destination, const unsigned char *mask, const size_t sz_x, const size_t sz_y, const int x, const int y, const int dx, const int dy, float& sum1, float& sum2, const size_t pos);

__global__ void perform_jacobi_iteration_cuda(const float * b1, float * b2, const unsigned char * source, const unsigned char * destination, 
const unsigned char *mask, const size_t sz_x, const size_t sz_y, const size_t * els_x_coords, const size_t * els_y_coords, const size_t numEls);

void perform_jacobi_iteration(const float * b1, float * b2, const unsigned char * source, const unsigned char *destination, 
				const unsigned char *mask, const Positions& pts, const size_t sz_x, const size_t sz_y);

float * perform_jacobi(const unsigned char * source, const unsigned char * destination, const unsigned char * mask, 
		const Positions& pts, const size_t sz_x, const size_t sz_y, const size_t num_iter);

__global__ void substitute_interior_pixels_cuda(uchar4 * d_destImg, const float * buffer_R, const float *buffer_G, const float * buffer_B, 
			const size_t * x_coords, const size_t * y_coords, const size_t num_coords, const size_t sz_x);

void substitute_interior_pixels(uchar4 * d_destImg, const float * buffer_R, const float *buffer_G, const float * buffer_B, 
			const unsigned char * d_refined_mask, const size_t sz, Positions& pts, const size_t sz_x);

class Positions
{
	public:
		Positions(const size_t num_pos) : m_length(num_pos)
		{
			checkCudaErrors(cudaMalloc(&m_x, sizeof(size_t) * num_pos));
			checkCudaErrors(cudaMalloc(&m_y, sizeof(size_t) * num_pos));
		} 

		~Positions()
		{
			if(m_x == nullptr) return;

			checkCudaErrors(cudaFree(m_x));
			m_x = nullptr;
			checkCudaErrors(cudaFree(m_y));
			m_y = nullptr;
			m_length = 0;
		}

		size_t * getX(){return m_x;}
		size_t * getY(){return m_y;}


		const size_t * getX()const {return m_x;}
		const size_t * getY()const {return m_y;}

		size_t getLength() const {return m_length;}
 
		Positions(const Positions&) = delete; 
		Positions& operator=(const Positions&) = delete;
		Positions(Positions&& other)
		{
			std::swap(m_x, other.m_x);
			std::swap(m_y, other.m_y);
			m_length = other.m_length;
		}

		Positions& operator=(Positions&& other)
		{
			if(this == &other) return *this;
			
			this->~Positions();

			std::swap(m_x, other.m_x);
			std::swap(m_y, other.m_y);
			m_length = other.m_length;		
			return *this;	
		} 

	private:
		size_t *m_x = nullptr;
		size_t *m_y = nullptr;	
		size_t m_length = 0;
};

/**
 * @brief Entry point function, provided by the HW. 
 *
 * This functions blend a source image with a destination image and fills an output buffer h_blendedImg
 * 
 * @param h_sourceImg:   input, source image (element to paste in the destination image), in host memory
 * @param numRowsSource: input, number of rows of the three images (source, destination, blended)
 * @param numColsSource: input, number of columns of the three images (source, destination, blended)
 * @param h_destImg:     input, destination image, in host memory
 * @param h_blendedImg:  output, blended image, in host memory


 */
void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

	uchar4 *d_sourceImg = NULL;
	uchar4 *d_destImg = NULL;

	checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4) * numRowsSource * numColsSource));

	checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyHostToDevice));


  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

	*/

	unsigned char *d_mask = NULL;
	checkCudaErrors(cudaMalloc(&d_mask, sizeof(unsigned char) * numRowsSource * numColsSource));
	unsigned char *d_refined_mask = NULL;
	checkCudaErrors(cudaMalloc(&d_refined_mask , sizeof(unsigned char) * numRowsSource * numColsSource));


	generate_mask(d_mask, d_sourceImg, numColsSource * numRowsSource);

/*     
	2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.
*/

	calculate_interior_border(d_mask, d_refined_mask, numColsSource, numRowsSource);
	checkCudaErrors(cudaFree(d_mask));

	//Obtains an array with all the elements in the "interior" used to launch only threads that actually do some work
	Positions pts = compact_interior_border(d_refined_mask, numColsSource, numRowsSource);

/*
     3) Separate out the incoming images into three separate channels
*/

  	auto extract_channel_R = [] __device__ (uchar4 el) -> unsigned char { return el.x; };
  	auto extract_channel_G = [] __device__ (uchar4 el) -> unsigned char { return el.y; };
  	auto extract_channel_B = [] __device__ (uchar4 el) -> unsigned char { return el.z; };
	
	unsigned char * d_sourceImg_R = NULL;
	unsigned char * d_sourceImg_G = NULL;
	unsigned char * d_sourceImg_B = NULL;

	checkCudaErrors(cudaMalloc(&d_sourceImg_R, sizeof(unsigned char) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMalloc(&d_sourceImg_G, sizeof(unsigned char) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMalloc(&d_sourceImg_B, sizeof(unsigned char) * numRowsSource * numColsSource));

	extract_channel(d_sourceImg_R, d_sourceImg, numRowsSource * numColsSource, extract_channel_R);
	extract_channel(d_sourceImg_G, d_sourceImg, numRowsSource * numColsSource, extract_channel_G);
	extract_channel(d_sourceImg_B, d_sourceImg, numRowsSource * numColsSource, extract_channel_B);
 
	unsigned char * d_destinationImg_R = NULL;
	unsigned char * d_destinationImg_G = NULL;
	unsigned char * d_destinationImg_B = NULL;

	checkCudaErrors(cudaMalloc(&d_destinationImg_R, sizeof(unsigned char) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMalloc(&d_destinationImg_G, sizeof(unsigned char) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMalloc(&d_destinationImg_B, sizeof(unsigned char) * numRowsSource * numColsSource));

	extract_channel(d_destinationImg_R, d_destImg, numRowsSource * numColsSource, extract_channel_R);
	extract_channel(d_destinationImg_G, d_destImg, numRowsSource * numColsSource, extract_channel_G);
	extract_channel(d_destinationImg_B, d_destImg, numRowsSource * numColsSource, extract_channel_B);
 

/*   4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.
*/

	float * buff_R = perform_jacobi(d_sourceImg_R, d_destinationImg_R, d_refined_mask, pts, numColsSource, numRowsSource, 800);
	float * buff_G = perform_jacobi(d_sourceImg_G, d_destinationImg_G, d_refined_mask, pts, numColsSource, numRowsSource, 800);
	float * buff_B = perform_jacobi(d_sourceImg_B, d_destinationImg_B, d_refined_mask, pts, numColsSource, numRowsSource, 800);

	substitute_interior_pixels(d_destImg, buff_R, buff_G, buff_B, d_refined_mask, numColsSource * numRowsSource, pts, numColsSource);

	checkCudaErrors(cudaMemcpy(h_blendedImg, d_destImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyDeviceToHost));

/*
     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

	checkCudaErrors(cudaFree(buff_R));
	checkCudaErrors(cudaFree(buff_G));
	checkCudaErrors(cudaFree(buff_B));

	checkCudaErrors(cudaFree(d_sourceImg_R));
	checkCudaErrors(cudaFree(d_sourceImg_G));
	checkCudaErrors(cudaFree(d_sourceImg_B));

	checkCudaErrors(cudaFree(d_destinationImg_R));
	checkCudaErrors(cudaFree(d_destinationImg_G));
	checkCudaErrors(cudaFree(d_destinationImg_B));

	checkCudaErrors(cudaFree(d_sourceImg));
	checkCudaErrors(cudaFree(d_refined_mask));
	checkCudaErrors(cudaFree(d_destImg));
}

/*----------------------------- Private Functions Implementation -----------------------------*/



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


template<typename device_scan_operator>
__global__ static void cuda_scan_in_block(const size_t * d_in, size_t * d_out, size_t * d_out_tails, const size_t length_vec, device_scan_operator op, unsigned int identity)
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

	if(d_out_tails != NULL && tid == blockDim.x - 1) 
	{
		d_out_tails[blockIdx.x] =  s_block_scan1[tid];
	}
}

template<typename device_scan_operator>
__global__ static  void cuda_scan_post_process(const size_t * in_vec, const size_t * in_vec_tails, size_t * out_vec, const size_t length_vec, device_scan_operator op, unsigned int cuda_scan_in_block_b_size, const size_t start)
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


template<typename operatorType>
static void scan(size_t * const d_vec, const size_t length, unsigned int identity_element, operatorType op, unsigned int start_element)
{

	static CudaBufferPool<size_t> pool(20);
	const int K = 256;
	const int K2 = 1024;
	const int num_blocks = std::max(1, static_cast<int>(std::ceil(length / (double)K)));
	const int num_blocks2 = std::max(1, static_cast<int>(std::ceil(length / (double)K2)));


	CudaBuffer<size_t> d_out_vect(pool.get(length));
	CudaBuffer<size_t> d_out_vec_tails(pool.get(num_blocks));
	
	cuda_scan_in_block<<<num_blocks, K, sizeof(size_t) * K * 2>>>(d_vec, d_out_vect.getDeviceBuffer(), d_out_vec_tails.getDeviceBuffer(), length, op, identity_element);
	checkCudaErrors(cudaGetLastError());

	if(num_blocks == 1)
	{
		cudaMemcpy(d_vec, d_out_vect.getDeviceBuffer(), length * sizeof(size_t), cudaMemcpyDeviceToDevice);
	}
	else
	{
		scan(d_out_vec_tails.getDeviceBuffer(), num_blocks, identity_element, op, identity_element);
		cuda_scan_post_process<<<num_blocks2, K2>>>(d_out_vect.getDeviceBuffer(), d_out_vec_tails.getDeviceBuffer(), d_vec, length, op, K, start_element);
	}

	checkCudaErrors(cudaGetLastError());		

	pool.add(std::move(d_out_vect));
	pool.add(std::move(d_out_vec_tails));
}

template<typename T1, typename T2>
void copy(T1 * dest, const T2 * source, std::size_t sz)
{
	const size_t num_threads = 128;
	size_t num_blocks = (size_t)std::ceil(sz / (double)num_threads);
	if(num_blocks == 0)  num_blocks = 1;
	
	auto op = []__device__(unsigned char in) -> float {return (float)in;};

	map<<<num_blocks, num_threads>>>(dest, source, sz, op); 
	checkCudaErrors(cudaGetLastError());
}



__global__ void cuda_compact(size_t * x_arr, size_t * y_arr, size_t * mask_scan, const size_t length_mask_compacted, const unsigned char * mask, const size_t sz_x, const size_t sz_y)
{
	size_t pos_x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t pos_y = threadIdx.y + blockDim.y * blockIdx.y;

	if(pos_x >= sz_x || pos_y >= sz_y) return;
	
	size_t pos = pos_y * sz_x + pos_x;

	if(mask[pos] != mask_region_interior) return;

	const size_t compact_pos = mask_scan[pos];

	x_arr[compact_pos] = pos_x;
	y_arr[compact_pos] = pos_y;
}

template<typename OType, typename IType, typename OperatorType>
__global__ void map(OType * out, const  IType * in, const size_t sz, OperatorType op)
{
	size_t pos = blockDim.x * blockIdx.x + threadIdx.x;
	if(pos >= sz) return;
	out[pos] = op(in[pos]);
}


Positions compact_interior_border(const unsigned char * d_refined_mask, const size_t numColsSource, const size_t numRowsSource)
{
	const size_t this_mark = mask_region_interior;	
	
	auto op_map = [this_mark]__device__ __host__(unsigned char e1)	-> size_t 
			{
				return e1 == this_mark;
			};

	const size_t l = numColsSource * numRowsSource;

	CudaBuffer<size_t> mask_indexes(l);
	
	const size_t num_threads = 512;
	const size_t num_blocks = std::max(1.0, std::ceil((double) l / num_threads));

	map<<<num_threads, num_blocks>>>(mask_indexes.getDeviceBuffer(), d_refined_mask, l, op_map);
	checkCudaErrors(cudaGetLastError());


	auto scan_op = []__device__ __host__(const size_t e1, const size_t e2)
			{	
				return e1 + e2;
			};

	scan(mask_indexes.getDeviceBuffer(), l, 0, scan_op, 0);
	checkCudaErrors(cudaGetLastError());

	int length_compact = 0;
	unsigned char last_el = 0; 
	checkCudaErrors(cudaMemcpy(&length_compact, mask_indexes.getDeviceBuffer() + l - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&last_el, d_refined_mask + l - 1, sizeof(unsigned char), cudaMemcpyDeviceToHost));
	
	if(last_el == mask_region_interior)
		length_compact++;

	Positions pos(length_compact);
	const size_t num_threads_x = 16;
	const size_t num_threads_y = 16;
	const size_t num_blocks_x =  std::max(1.0, std::ceil((double) numColsSource / num_threads_x));
	const size_t num_blocks_y =  std::max(1.0, std::ceil((double) numRowsSource / num_threads_y));

	cuda_compact<<<dim3(num_blocks_x,  num_blocks_y), dim3(num_threads_x,num_threads_y)>>>(pos.getX(), pos.getY(), mask_indexes.getDeviceBuffer(), mask_indexes.getNumEls(), d_refined_mask, numColsSource, numRowsSource);
	checkCudaErrors(cudaGetLastError());
	return pos;
}


void generate_mask(unsigned char * mask, const uchar4 * img, const size_t sz)
{
	const size_t num_threads = 128;
	size_t num_blocks = (size_t)std::ceil(sz / (double)num_threads);
	if(num_blocks == 0)  num_blocks = 1;
	
	auto op =  []__device__(const uchar4 el) -> unsigned char
	{
		if(el.x == el.y && el.y == el.z && el.z == 255) return non_mask_region;
		return mask_region;
	};
	map<<<num_blocks, num_threads>>>(mask, img, sz, op);
	checkCudaErrors(cudaGetLastError());
}   


template<typename opType>
void extract_channel(unsigned char * out_img, const uchar4 * img, const size_t sz, opType op)
{
	const size_t num_threads = 128;
	size_t num_blocks = (size_t)std::ceil(sz / (double)num_threads);
	if(num_blocks == 0)  num_blocks = 1;
	
	map<<<num_blocks, num_threads>>>(out_img, img, sz, op);
	checkCudaErrors(cudaGetLastError());
}   

__device__ void add_neigh(const unsigned char * in_mask, const size_t x_sz, const size_t y_sz, const int x, const int y, const int dx, const int dy, int &num_valid_neigh, int &num_masked_neigh)
{
	int yn = y + dy;
	int xn = x + dx;

	if(xn < 0 || xn >= x_sz) return;
	if(yn < 0 || yn >= y_sz) return;

	num_valid_neigh++;
	num_masked_neigh += in_mask[yn * x_sz + xn] == mask_region;	
}

__global__ void find_border_interior(const unsigned char * in_mask, unsigned char * out_mask, const size_t x_sz, const size_t y_sz)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x >= x_sz) return;
	if(y >= y_sz) return;

	int pos = y * x_sz + x;

	if(in_mask[pos] == non_mask_region) 
	{
		out_mask[pos] = non_mask_region;
		return;
	}

	int num_valid_neigh = 0;
	int num_masked_neigh = 0;

	add_neigh(in_mask, x_sz, y_sz, x, y,  0,  1, num_valid_neigh, num_masked_neigh);
	add_neigh(in_mask, x_sz, y_sz, x, y,  0, -1, num_valid_neigh, num_masked_neigh);
	add_neigh(in_mask, x_sz, y_sz, x, y, -1,  0, num_valid_neigh, num_masked_neigh);
	add_neigh(in_mask, x_sz, y_sz, x, y,  1,  0, num_valid_neigh, num_masked_neigh);

	if(num_masked_neigh == num_valid_neigh &&  num_valid_neigh > 0)
		out_mask[pos] = mask_region_interior;
	else
		out_mask[pos] = mask_region_border;
	
}

void calculate_interior_border(const unsigned char * in_mask, unsigned char * out_mask, const size_t x_sz, const size_t y_sz)
{
	const size_t num_threads = 32;
	size_t num_blocks_x = (size_t)std::ceil(x_sz / (double)num_threads);	
	size_t num_blocks_y = (size_t)std::ceil(y_sz / (double)num_threads);
	find_border_interior<<<dim3(num_blocks_x,  num_blocks_y), dim3(num_threads,num_threads)>>>(in_mask, out_mask, x_sz, y_sz);
	checkCudaErrors(cudaGetLastError());
}

__device__ void perform_j_neigh(const float * b1, const unsigned char * source, const unsigned char * destination, const unsigned char *mask, const size_t sz_x, const size_t sz_y, const int x, const int y, const int dx, const int dy, float& sum1, float& sum2, const size_t pos)
{
	int xn = x + dx;
	int yn = y + dy;

	if(xn < 0 || xn >= sz_x) return;
	if(yn < 0 || yn >= sz_y) return;


	int neigh = yn * sz_x + xn;
	if(mask[neigh] == mask_region_interior)
	{
		sum1 += (float)b1[neigh];
	}
	else
	{
		sum1 += (float)destination[neigh];
	}

	sum2 += (float)source[pos] - (float)source[neigh];	
}

__global__ void perform_jacobi_iteration_cuda(const float * b1, float * b2, const unsigned char * source, const unsigned char * destination, 
const unsigned char *mask, const size_t sz_x, const size_t sz_y, const size_t * els_x_coords, const size_t * els_y_coords, const size_t numEls)
{

/*
	1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
	Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
		else if the neighbor in on the border then += DestinationImg[neighbor]

	Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

	2) Calculate the new pixel value:
	float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
	ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]
*/

	const size_t pos = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(pos >= numEls) return;
	
	const int x = els_x_coords[pos];
	const int y = els_y_coords[pos];

	float sum1 = 0.0f;
	float sum2 = 0.0f;

	const int pos_dst = x + y * (int)sz_x;

	perform_j_neigh(b1, source, destination, mask, sz_x, sz_y, x, y,  1,  0, sum1, sum2, pos_dst);
	perform_j_neigh(b1, source, destination, mask, sz_x, sz_y, x, y, -1,  0, sum1, sum2, pos_dst);
	perform_j_neigh(b1, source, destination, mask, sz_x, sz_y, x, y,  0, -1, sum1, sum2, pos_dst);
	perform_j_neigh(b1, source, destination, mask, sz_x, sz_y, x, y,  0,  1, sum1, sum2, pos_dst);

	float newVal = (sum1 + sum2) / 4.0f;
	newVal = min(255.0f, max(0.0f, newVal));	

	b2[pos_dst] = newVal;
}

void perform_jacobi_iteration(const float * b1, float * b2, const unsigned char * source, const unsigned char *destination, 
				const unsigned char *mask, const Positions& pts, const size_t sz_x, const size_t sz_y)
{
	const size_t num_threads = 128;
	const size_t l = pts.getLength();
	const size_t num_blocks = std::max(1.0, std::ceil(l / (double)num_threads));
	

	perform_jacobi_iteration_cuda<<<num_threads, num_blocks>>>(b1, b2, source, destination, mask, sz_x, sz_y, pts.getX(), pts.getY(), pts.getLength());
	checkCudaErrors(cudaGetLastError());
}

float * perform_jacobi(const unsigned char * source, const unsigned char * destination, const unsigned char * mask, 
		const Positions& pts, const size_t sz_x, const size_t sz_y, const size_t num_iter)
{
	float * b1 = NULL;
	float * b2 = NULL;

	checkCudaErrors(cudaMalloc(&b1, sizeof(float) * sz_x * sz_y));
	checkCudaErrors(cudaMalloc(&b2, sizeof(float) * sz_x * sz_y));

	copy(b1, source, sz_x * sz_y);

	for(size_t i = 0; i < num_iter; ++i)
	{
		perform_jacobi_iteration(b1, b2, source, destination, mask, pts, sz_x, sz_y);
		std::swap(b1, b2);
	}
	
	checkCudaErrors(cudaFree(b2));
	return b1;
}

__global__ void substitute_interior_pixels_cuda(uchar4 * d_destImg, const float * buffer_R, const float *buffer_G, const float * buffer_B, 
			const size_t * x_coords, const size_t * y_coords, const size_t num_coords, const size_t sz_x)
{
	/*
	 6) Create the output image by replacing all the interior pixels
	in the destination image with the result of the Jacobi iterations.
	Just cast the floating point values to unsigned chars since we have
	already made sure to clamp them to the correct range.
	*/	
	
	const size_t pos =  blockDim.x * blockIdx.x + threadIdx.x;

	if(pos >= num_coords) return;

	const size_t x = x_coords[pos];
	const size_t y = y_coords[pos];
	const size_t dest_pos = x + y * sz_x;

	d_destImg[dest_pos].x = (unsigned char) buffer_R[dest_pos];
	d_destImg[dest_pos].y = (unsigned char) buffer_G[dest_pos];
	d_destImg[dest_pos].z = (unsigned char) buffer_B[dest_pos];
}

void substitute_interior_pixels(uchar4 * d_destImg, const float * buffer_R, const float *buffer_G, const float * buffer_B, 
			const unsigned char * d_refined_mask, const size_t sz, Positions& pts, const size_t sz_x)
{
	const size_t num_threads = 256;
	size_t num_blocks = (size_t)std::ceil(pts.getLength() / (double)num_threads);
	if(num_blocks == 0)  num_blocks = 1;
	
	substitute_interior_pixels_cuda<<<num_blocks, num_threads>>>(d_destImg, buffer_R, buffer_G, buffer_B, pts.getX(), pts.getY(), pts.getLength(), sz_x); 
	checkCudaErrors(cudaGetLastError());
}
