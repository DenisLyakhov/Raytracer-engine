#include <iostream>
#include <curand_kernel.h>
#include <math.h>
#include <chrono>

#include "Raytracer.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
	JsonParser::loadJsonObjectConfigs();

	int x = 0;
	int y = 0;

	if (argc > 2) {
		x = atoi(argv[1]);
		y = atoi(argv[2]);
	}

	int sceneConfig = 1;

	Raytracer::initialize(x, y);

	Raytracer::renderImage(sceneConfig);

	std::clock_t start;
	start = std::clock();

	Raytracer::renderImage(sceneConfig);

	std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

	delete[] objectConfigs;
	return 0;
}