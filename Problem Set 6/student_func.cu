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
		if(el.x == el.y && el.y == el.z && el.z == 255) return 255;
		return 0;
	};
	map<<<num_blocks, num_threads>>>(mask, img, sz, op);
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


  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

	*/

	unsigned char *d_mask = NULL;
	checkCudaErrors(cudaMalloc(&d_mask, sizeof(unsigned char) * numRowsSource * numColsSource));
	generate_mask(d_mask, d_sourceImg, numRowsSource * numColsSource);
	
/*     
	2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

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

	cudaFree(d_sourceImg);
	cudaFree(d_mask);
	cudaFree(d_destImg);
}
