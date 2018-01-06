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

const unsigned char non_mask_region = 0;
const unsigned char mask_region = 1; //unknown if border or or border
const unsigned char mask_region_interior = 2; 
const unsigned char mask_region_border = 3;
 
template<typename OType, typename IType, typename OperatorType>
__global__ void map(OType * out, const  IType * in, const size_t sz, OperatorType op)
{
	size_t pos = blockDim.x * blockIdx.x + threadIdx.x;
	if(pos >= sz) return;
	out[pos] = op(in[pos]);
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

__device__ void perform_j_neigh(const float * b1, const unsigned char * source, const unsigned char * destination, const unsigned char *mask, const size_t sz_x, const size_t sz_y, const int x, const int y, const int dx, const int dy, float& sum1, float& sum2,
const int pos)
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

__global__ void perform_jacobi_iteration_cuda(const float * b1, float * b2, const unsigned char * source, const unsigned char * destination, const unsigned char *mask, const size_t sz_x, const size_t sz_y)
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

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x >= sz_x) return;
	if(y >= sz_y) return;

	int pos = y * sz_x + x;

	if(mask[pos] != mask_region_interior) return;

	float sum1 = 0.0f;
	float sum2 = 0.0f;

	perform_j_neigh(b1, source, destination, mask, sz_x, sz_y, x, y,  1,  0, sum1, sum2, pos);
	perform_j_neigh(b1, source, destination, mask, sz_x, sz_y, x, y, -1,  0, sum1, sum2, pos);
	perform_j_neigh(b1, source, destination, mask, sz_x, sz_y, x, y,  0, -1, sum1, sum2, pos);
	perform_j_neigh(b1, source, destination, mask, sz_x, sz_y, x, y,  0,  1, sum1, sum2, pos);

	float newVal = (sum1 + sum2) / 4.0f;
	newVal = min(255.0f, max(0.0f, newVal));	
	b2[pos] = newVal;
}

void perform_jacobi_iteration(const float * b1, float * b2, const unsigned char * source, const unsigned char *destination, 
				const unsigned char *mask, const size_t sz_x, const size_t sz_y)
{
	const size_t num_threads = 32;
	size_t num_blocks_x = (size_t)std::ceil(sz_x / (double)num_threads);	
	size_t num_blocks_y = (size_t)std::ceil(sz_y / (double)num_threads);

	perform_jacobi_iteration_cuda<<<dim3(num_blocks_x,  num_blocks_y), dim3(num_threads,num_threads)>>>(b1, b2, source, destination, mask, sz_x, sz_y);
	checkCudaErrors(cudaGetLastError());
}

float * perform_jacobi(const unsigned char * source, const unsigned char * destination, const unsigned char * mask, 
		const size_t sz_x, const size_t sz_y, const size_t num_iter)
{
	float * b1 = NULL;
	float * b2 = NULL;

	checkCudaErrors(cudaMalloc(&b1, sizeof(float) * sz_x * sz_y));
	checkCudaErrors(cudaMalloc(&b2, sizeof(float) * sz_x * sz_y));

	copy(b1, source, sz_x * sz_y);

	for(size_t i = 0; i < num_iter; ++i)
	{
		perform_jacobi_iteration(b1, b2, source, destination, mask, sz_x, sz_y);
		std::swap(b1, b2);
	}
	
	checkCudaErrors(cudaFree(b2));
	return b1;
}

__global__ void substitute_interior_pixels_cuda(uchar4 * d_destImg, const float * buffer_R, const float *buffer_G, const float * buffer_B, 
			const unsigned char * mask, const size_t sz)
{
	/*
	 6) Create the output image by replacing all the interior pixels
	in the destination image with the result of the Jacobi iterations.
	Just cast the floating point values to unsigned chars since we have
	already made sure to clamp them to the correct range.
	*/	
	
	size_t pos =  blockDim.x * blockIdx.x + threadIdx.x;

	if(pos >= sz) return;

	if(mask[pos] != mask_region_interior) return;

	d_destImg[pos].x = (unsigned char) buffer_R[pos];
	d_destImg[pos].y = (unsigned char) buffer_G[pos];
	d_destImg[pos].z = (unsigned char) buffer_B[pos];
}

void substitute_interior_pixels(uchar4 * d_destImg, const float * buffer_R, const float *buffer_G, const float * buffer_B, 
			const unsigned char * d_refined_mask, const size_t sz)
{
	const size_t num_threads = 128;
	size_t num_blocks = (size_t)std::ceil(sz / (double)num_threads);
	if(num_blocks == 0)  num_blocks = 1;
	
	substitute_interior_pixels_cuda<<<num_blocks, num_threads>>>(d_destImg, buffer_R, buffer_G, buffer_B, d_refined_mask, sz); 
	checkCudaErrors(cudaGetLastError());
}


void print_arr_uchar(const uchar4 * arr, size_t length)
{
	std::vector<uchar4> h_arr(length);
	cudaMemcpy(h_arr.data(), arr, sizeof(uchar4) * length, cudaMemcpyDeviceToHost);

	for(const auto e : h_arr)
		std::cout<<(float)e.x<<"|"<<(float)e.x<<"|"<<(float)e.z<<"\n";
	std::cout<<"-------------------------\n";
}

template<typename T>
void print_arr(const T * arr, size_t length)
{
	std::vector<T> h_arr(length);
	cudaMemcpy(h_arr.data(), arr, sizeof(T) * length, cudaMemcpyDeviceToHost);

	for(int i = 0; i < h_arr.size(); ++i)
		std::cout<<i<<"|"<<(float)h_arr[i]<<"\n";
	std::cout<<"-------------------------\n";
}

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

	float * buff_R = perform_jacobi(d_sourceImg_R, d_destinationImg_R, d_refined_mask, numColsSource, numRowsSource, 800);
	float * buff_G = perform_jacobi(d_sourceImg_G, d_destinationImg_G, d_refined_mask, numColsSource, numRowsSource, 800);
	float * buff_B = perform_jacobi(d_sourceImg_B, d_destinationImg_B, d_refined_mask, numColsSource, numRowsSource, 800);

	substitute_interior_pixels(d_destImg, buff_R, buff_G, buff_B, d_refined_mask, numColsSource * numRowsSource);

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
