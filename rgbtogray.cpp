#include "loadpng.h"

//Read image
std::vector<unsigned char> image;
unsigned int width,height;

unsigned error = loadpng::decode(image,width,height,input_file);

std::vector<unsigned char> out_image(image.size(),255);
error = loadpng::encode(output_file,out_image,width,height)