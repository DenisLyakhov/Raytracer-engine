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

	int sceneConfig = 1;

	if (argc > 2) {
		x = atoi(argv[1]);
		y = atoi(argv[2]);
		if (argc > 3) {
			sceneConfig = atoi(argv[3]);
		}
	}

	Raytracer::initialize(x, y, sceneConfig);

	Raytracer::renderImage(sceneConfig);

	delete[] objectConfigs;
	return 0;
}