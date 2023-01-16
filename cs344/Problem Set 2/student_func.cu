// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "utils.h"

__global__
void gaussian_blur(const uchar4 *const input_RGBA, uchar4 *const out_RGBA, const int numRows, const int numCols, 
                   const float *const filter, const int filter_width) {
  const int t_id_x = threadIdx.x + (blockDim.x * blockIdx.x);
  const int t_id_y = threadIdx.y + (blockDim.y * blockIdx.y);
  const int stride_x = blockDim.x * gridDim.x;
  const int stride_y = blockDim.y * gridDim.y;
  const int half_filter_width = filter_width >> 1;
  
  for (int y = t_id_y; y < numCols; y += stride_y) {
    for (int x = t_id_x; x < numRows; x += stride_x) {

      float con_val[3] = {0.0f, 0.0f, 0.0f};
      for (int filter_y = -half_filter_width; filter_y <= half_filter_width; filter_y++) {
        for (int filter_x = -half_filter_width; filter_x <= half_filter_width; filter_x++) {
          const int pix_x = min(max(x + filter_x, 0), static_cast<int>(numRows-1));
          const int pix_y = min(max(y + filter_y, 0), static_cast<int>(numCols-1));
          const int kern_x = filter_x + half_filter_width;
          const int kern_y = filter_y + half_filter_width;
          const uchar4& pix_val =  input_RGBA[pix_x * numCols + pix_y];
          const float filter_val = filter[kern_x * filter_width + kern_y];
          con_val[0] += static_cast<float>(pix_val.x) * filter_val;
          con_val[1] += static_cast<float>(pix_val.y) * filter_val;
          con_val[2] += static_cast<float>(pix_val.z) * filter_val;
        }
      }
      const uchar4 rgba_pix_val = {static_cast<uint8_t>(con_val[0]), static_cast<uint8_t>(con_val[1]), static_cast<uint8_t>(con_val[2]), 255};
      out_RGBA[x * numCols + y] = rgba_pix_val;
    }
  }
  return;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  //checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  //checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  //checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  #if 0
  int no_Of_Threads_x = 32;
  int no_Of_Blks_x_Per_Grid   = numRows / no_Of_Threads_x;
  no_Of_Blks_x_Per_Grid = (0 == numRows % no_Of_Blks_x_Per_Grid) ? no_Of_Blks_x_Per_Grid : no_Of_Blks_x_Per_Grid+1;

  int no_Of_Threads_y = 32;
  int no_Of_Blks_y_Per_Grid   = numCols / no_Of_Threads_y;
  no_Of_Blks_y_Per_Grid = (0 == numCols % no_Of_Blks_y_Per_Grid) ? no_Of_Blks_y_Per_Grid : no_Of_Blks_y_Per_Grid+1;

#else
  int no_Of_Threads_x = 16;
  int no_Of_Blks_x_Per_Grid  = 32;

  int no_Of_Threads_y = 16;
  int no_Of_Blks_y_Per_Grid   = 32;
  
#endif
  //TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(no_Of_Threads_x, no_Of_Threads_y);

  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize( no_Of_Blks_x_Per_Grid, no_Of_Blks_y_Per_Grid);
  gaussian_blur<<<gridSize,  blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, d_filter, filterWidth);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
#if 0  
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
#endif  
  checkCudaErrors(cudaFree(d_filter));
}
