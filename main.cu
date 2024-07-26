#include <iostream>
#include <curand_kernel.h>
#include <math.h>

#include "Raytracer.h"

#define RAYS_PER_PIXEL 25
#define MAX_BOUNCES_PER_RAY 10

#define GRID_WIDTH 16

using namespace std;

int main(int argc, char* argv[]) {
	JsonParser::loadJsonObjectConfigs();

	int x = 0;
	int y = 0;

	if (argc > 2) {
		x = atoi(argv[1]);
		y = atoi(argv[2]);
	}

	Raytracer::initialize(x, y);
	Raytracer::renderImage();

	delete[] objectConfigs;
	return 0;
}