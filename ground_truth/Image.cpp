#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

class Image{
    public:
        Image(string filename);
        float *rgb;
        int width, height, channels;
};

Image::Image(string filename)
{
    unsigned char* data = stbi_load(filename.c_str(), &this->width, &this->height, &this->channels, 0);
    this->rgb = new float[(this->width + 50) * (this->height + 50)];
    fill_n(this->rgb, (this->width + 50) * (this->height + 50), 0);
    
    for (int i = 0; i < this->height; i++){
        for (int j = 0; j < this->width; j++){
            int index = this->channels * (i * this->width + j);
            int r = (int)data[index + 0];
            int g = (int)data[index + 1];
            int b = (int)data[index + 2];
            float value = 0.2989 * r + 0.5870 * g + 0.1140 * b;
            this->rgb[i * this->width + j] = value;
        }
    }
}

