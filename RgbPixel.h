#pragma once
#include <sstream>
#include <math.h>

using namespace std;

class RgbPixel {
public:
	float r;
	float g;
	float b;

	__host__ __device__ RgbPixel() {
		r = 0.0;
		g = 0.0;
		b = 0.0;
	}

	__host__ __device__ RgbPixel(float r, float g, float b) {
		this->r = r;
		this->g = g;
		this->b = b;
	}

	__host__ __device__ inline RgbPixel operator+(RgbPixel pixel) {
		return RgbPixel(
			this->r + pixel.r,
			this->g + pixel.g,
			this->b + pixel.b);
	}

	__host__ __device__ inline RgbPixel operator*(RgbPixel pixel) {
		return RgbPixel(
			this->r * pixel.r,
			this->g * pixel.g,
			this->b * pixel.b);
	}

	__host__ __device__ inline RgbPixel operator*(float t) {
		return RgbPixel(
			this->r * t,
			this->g * t,
			this->b * t);
	}

	__host__ __device__ inline RgbPixel operator/(float t) {
		return RgbPixel(
			this->r / t,
			this->g / t,
			this->b / t);
	}

	__host__ void writeToImage(stringstream& outputPixel) {
		const int colorRange = 255;

		// Translate color range [0, 1] to [0, 255]
		int r = this->r * colorRange;
		int g = this->g * colorRange;
		int b = this->b * colorRange;

		outputPixel << r << ' ' << g << ' ' << b << '\n';
	}
};
