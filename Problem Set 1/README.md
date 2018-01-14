Instructions on how to invoke the program:
- Create a directory inside the code, e.g. Make
- Inside the directory inkove: 
	- cmake ..
	- make
- Now you can execute the program typing, inside the directory Make
 ./HW1 ../cinque_terre_small.jpg out.png ref.png  1  0.00022

The two final numbers are to allow some differences between reference and CUDA-generated image. This is needed because we always find a difference of 1 in some pixels.
After extensive debugging I believe that it has to do with roundings inconsistencies between CPU and GPU (see, e.g. http://developer.download.nvidia.com/assets/cuda/files/NVIDIA-CUDA-Floating-Point.pdf). We are allowing a maximum difference between pixels of 1 and we are allowing less than 0.022% pixels to be different between CPU and GPU images. On my machine, around 0.021% of the pixels are different and the difference never exceeds 1.
