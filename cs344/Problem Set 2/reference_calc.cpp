#include <algorithm>
#include <cassert>
// for uchar4 struct
#include <cuda_runtime.h>

void channelConvolution(const unsigned char* const inputChannel,
                        unsigned char* const channelBlurred,
                        const size_t numRows, const size_t numCols,
                        const float *filter, const int filterWidth)
{
  //Dealing with an even width filter is trickier
  assert(filterWidth % 2 == 1);
#if 0
  //For every pixel in the image
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      float result = 0.f;
      //For every value in the filter around the pixel (c, r)
      for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
          //Find the global image position for this filter position
          //clamp to boundary of the image
		  int image_r = std::min(std::max(r + filter_r, 0), static_cast<int>(numRows - 1));
          int image_c = std::min(std::max(c + filter_c, 0), static_cast<int>(numCols - 1));

          float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
          float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

          result += image_value * filter_value;
        }
      }

      channelBlurred[r * numCols + c] = result;
    }
  }
#else
  int t_id_x = 0;
  int stride_x = 1;
  int t_id_y = 0;
  int stride_y = 1;
  const int half_filt_width = filterWidth / 2;

  for (int image_y = t_id_y; image_y < numCols; image_y += stride_y) {
    for (int image_x = t_id_x; image_x < numRows; image_x += stride_x) {
      float fconv_pix_val = 0.0f;

      for (int kern_y = -half_filt_width; kern_y <= half_filt_width; kern_y++) {
        for (int kern_x = -half_filt_width; kern_x <= half_filt_width; kern_x++) {
          int input_x = std::min(std::max(image_x+kern_x , 0) , static_cast<int>(numRows - 1));
          int input_y = std::min(std::max(image_y+kern_y , 0) , static_cast<int>(numCols - 1));
          float filter_val = filter[(half_filt_width + kern_x) * filterWidth + kern_y + half_filt_width];
          float pix_val    = static_cast<float>(inputChannel[input_y + (input_x * numCols)]);
          fconv_pix_val += (filter_val * pix_val);
        }
      }
      channelBlurred[image_y + (image_x * numCols)] = fconv_pix_val;
    }  
  }
#endif  
}

void referenceCalculation(const uchar4* const rgbaImage, uchar4 *const outputImage,
                          size_t numRows, size_t numCols,
                          const float* const filter, const int filterWidth)
{
  unsigned char *red   = new unsigned char[numRows * numCols];
  unsigned char *blue  = new unsigned char[numRows * numCols];
  unsigned char *green = new unsigned char[numRows * numCols];

  unsigned char *redBlurred   = new unsigned char[numRows * numCols];
  unsigned char *blueBlurred  = new unsigned char[numRows * numCols];
  unsigned char *greenBlurred = new unsigned char[numRows * numCols];

  //First we separate the incoming RGBA image into three separate channels
  //for Red, Green and Blue
  for (size_t i = 0; i < numRows * numCols; ++i) {
    uchar4 rgba = rgbaImage[i];
    red[i]   = rgba.x;
    green[i] = rgba.y;
    blue[i]  = rgba.z;
  }

  //Now we can do the convolution for each of the color channels
  channelConvolution(red, redBlurred, numRows, numCols, filter, filterWidth);
  channelConvolution(green, greenBlurred, numRows, numCols, filter, filterWidth);
  channelConvolution(blue, blueBlurred, numRows, numCols, filter, filterWidth);

  //now recombine into the output image - Alpha is 255 for no transparency
  for (size_t i = 0; i < numRows * numCols; ++i) {
    uchar4 rgba = make_uchar4(redBlurred[i], greenBlurred[i], blueBlurred[i], 255);
    outputImage[i] = rgba;
  }

  delete[] red;
  delete[] green;
  delete[] blue;

  delete[] redBlurred;
  delete[] greenBlurred;
  delete[] blueBlurred;
}
