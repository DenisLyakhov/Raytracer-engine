#include <iostream>
#include <fstream>
#include <sstream>
#include <curand_kernel.h>
#include <math.h>

#include "RgbPixel.h"
#include "Ray.h"
#include "Lambertian.h"
#include "Metal.h"
#include "Glass.h"
#include "Sphere.h"
#include "Cube.h"
#include "RayHitRecord.h"
#include "JsonParser.h"

#define RAYS_PER_PIXEL 500
#define MAX_BOUNCES_PER_RAY 10

#define GRID_WIDTH 16

using namespace std;

// Image dimensions
const int displayWidth = 640;
const int displayHeight = 480;

// Camera configuration
float verticalFov = 40.f / 180 * 3.1415;
float viewHeight = 2.0 * tan(verticalFov);
float viewWidth = viewHeight * displayWidth / displayHeight;

Vector cameraPos = Vector(-2, 2, 1);
Vector lookAt = Vector(0, 0, -1);
Vector up = Vector(0, 1, 0);

Vector dir = (cameraPos - lookAt).getNormalizedVector();
Vector w = up.crossProduct(dir);
Vector h = dir.crossProduct(w);

Vector lowerRightPixelPos = w * viewWidth;
Vector upperLeftPixelPos = h * viewHeight;

Vector lowerLeftPixelPos = cameraPos - lowerRightPixelPos / 2 - upperLeftPixelPos / 2 - dir;


void writeImageToFile(string outputImage) {
	ofstream out("output_image.ppm");
	out << outputImage;
	out.close();
}

__device__ RgbPixel getBackgroundPixel(Ray ray) {
	return RgbPixel(1, 1, 1);
}

// Iterate through objects, get closest ray hit point
__device__ RayHitRecord getClosestRayHit(Ray ray, Object** sceneObjects, int objectsCount) {
	RayHitRecord closest;
	float closestT = INFINITY;

	for (int i = 0; i < objectsCount; i++) {
		Object* obj = sceneObjects[i];
		RayHitRecord hitRecord = obj->hitObject(ray);
		if (hitRecord.hit) {
			if (!closest.hit || hitRecord.t < closestT) {
				closestT = hitRecord.t;
				closest = hitRecord;
			}
		}
	}

	return closest;
}

__global__ void renderPixel(
	float* result,
	float viewWidth, float viewHeight,
	int displayWidth, int displayHeight,
	Vector cameraPos, Vector lowerLeftPixelPos,
	Vector lowerRightPixelPos, Vector upperLeftPixelPos,
	Object** objects, int objectsCount
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= displayWidth || y >= displayHeight) {
		return;
	}

	int pixelIndex = x * displayHeight + y;

	curandState localRandState;
	curand_init(13, pixelIndex, 0, &localRandState);

	result[3 * pixelIndex] = 0;
	result[3 * pixelIndex + 1] = 0;
	result[3 * pixelIndex + 2] = 0;

	for (int i = 0; i < RAYS_PER_PIXEL; i++) {
		float currentXPos = (float(x) + randomFloat(&localRandState)) / (displayWidth - 1);
		float currentYPos = (float(y) + randomFloat(&localRandState)) / (displayHeight - 1);

		Vector currentImagePos = lowerLeftPixelPos +
			lowerRightPixelPos * currentXPos +
			upperLeftPixelPos * currentYPos;

		Ray ray = Ray(cameraPos, currentImagePos - cameraPos);

		RgbPixel color = getBackgroundPixel(ray);

		for (int i = 0; i < MAX_BOUNCES_PER_RAY; i++) {
			RayHitRecord hitRecord = getClosestRayHit(ray, objects, objectsCount);
			if (hitRecord.hit) {
				ray = hitRecord.scatter(ray, &localRandState);
				color = color * hitRecord.material->color;
			}
			else {
				break;
			}
		}

		result[3 * pixelIndex] += color.r;
		result[3 * pixelIndex + 1] += color.g;
		result[3 * pixelIndex + 2] += color.b;
	}

	result[3 * pixelIndex] /= RAYS_PER_PIXEL;
	result[3 * pixelIndex + 1] /= RAYS_PER_PIXEL;
	result[3 * pixelIndex + 2] /= RAYS_PER_PIXEL;
}

__global__ void createWorld(Object** sceneObjects, ObjectConfig* objectConfigs, int objectsCount) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= objectsCount) return;

	ObjectConfig config = objectConfigs[i];

	Material* material;
	switch (config.material) {
	case GLASS:
		material = new Glass(RgbPixel(config.r, config.g, config.b), config.refractiveIndex);
		break;

	case LAMBERTIAN:
		material = new Lambertian(RgbPixel(config.r, config.g, config.b));
		break;

	case METAL:
		material = new Metal(RgbPixel(config.r, config.g, config.b), config.fuzziness);
		break;

	case DEFAULT:
		material = new Material(RgbPixel(config.r, config.g, config.b));
		break;
	}

	switch (config.type) {
	case SPHERE:
		sceneObjects[i] = new Sphere(Vector(config.x, config.y, config.z), config.radius, material);
		break;
	case CUBE:
		sceneObjects[i] = new Cube(Vector(config.x, config.y, config.z), config.radius, material);
		break;
	}
}

__global__ void freeWorld(Object** sceneObjects, int objectsCount) {
	for (int i = 0; i < objectsCount; i++) {
		delete* (sceneObjects + i);
	}
}

void renderImage() {

	// create scene objects on device
	Object** sceneObjects;
	ObjectConfig* devObjectConfigs;

	cudaMalloc((void**)&sceneObjects, objectsCount * sizeof(Object*));
	cudaMalloc((void**)&devObjectConfigs, objectsCount * sizeof(ObjectConfig));
	cudaMemcpy(devObjectConfigs, objectConfigs, objectsCount * sizeof(ObjectConfig), cudaMemcpyHostToDevice);

	createWorld << <1, objectsCount >> > (sceneObjects, devObjectConfigs, objectsCount);
	cudaDeviceSynchronize();

	// define thread grid dimensions
	dim3 blocks((displayWidth - 1) / GRID_WIDTH + 1, (displayHeight - 1) / GRID_WIDTH + 1);
	dim3 threads(GRID_WIDTH, GRID_WIDTH);

	// allocate result array
	float* result = new float[3 * displayWidth * displayHeight];
	float* devResult;
	cudaMalloc(&devResult, 3 * displayWidth * displayHeight * sizeof(float));

	// run parallel raytracing
	renderPixel << <blocks, threads >> > (devResult, viewWidth, viewHeight, displayWidth, displayHeight, cameraPos, lowerLeftPixelPos,
		lowerRightPixelPos, upperLeftPixelPos, sceneObjects, objectsCount);
	cudaMemcpy(result, devResult, 3 * displayWidth * displayHeight * sizeof(float), cudaMemcpyDeviceToHost);

	// write result image to file
	stringstream outputImage;
	outputImage << "P3\n" << displayWidth << ' ' << displayHeight << "\n255\n";

	for (int y = displayHeight - 1; y >= 0; y--) {
		for (int x = 0; x < displayWidth; x++) {
			RgbPixel pixel = RgbPixel(
				result[x * displayHeight * 3 + y * 3],
				result[x * displayHeight * 3 + y * 3 + 1],
				result[x * displayHeight * 3 + y * 3 + 2]
			);
			pixel.writeToImage(outputImage);
		}
	}
	writeImageToFile(outputImage.str());

	freeWorld << <1, 1 >> > (sceneObjects, objectsCount);
	cudaDeviceSynchronize();

	cudaFree(sceneObjects);
	cudaFree(devResult);
	delete[] result;
}

int main() {

	JsonParser::loadJsonObjectConfigs();

	renderImage();

	delete[] objectConfigs;
	return 0;
}