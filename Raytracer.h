#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <curand_kernel.h>
#include <math.h>
#include <cstdlib>

#include "RgbPixel.h"
#include "Ray.h"
#include "Lambertian.h"
#include "Metal.h"
#include "Glass.h"
#include "Sphere.h"
#include "RayHitRecord.h"
#include "JsonParser.h"

#define RAYS_PER_PIXEL 50
#define MAX_BOUNCES_PER_RAY 10

#define GRID_WIDTH 16

namespace Raytracer {
	// Image dimensions

	/*int displayWidth = 1280;
	int displayHeight = 720;*/

	int displayWidth = 640;
	int displayHeight = 360;

	/*int displayWidth = 2560;
	int displayHeight = 1440;*/

	//int displayWidth = 160;
	//int displayHeight = 90;

	// Camera configuration;
	float viewHeight;
	float viewWidth;

	//Vector cameraPos = Vector(13, 2, 3);
	Vector cameraPos = Vector(-4, 1.5, 2);

	Vector lowerRightPixelPos;
	Vector upperLeftPixelPos;

	Vector lowerLeftPixelPos;

	// Setup dimensions
	__host__ void initialize(int x, int y) {
		if (x != 0 && y != 0) {
			displayWidth = x;
			displayHeight = y;
		}

		cameraPos = Vector(cameraConfig.x, cameraConfig.y, cameraConfig.z);

		//Vector lookAt = Vector(-20, -5, 10);
		//Vector lookAt = Vector(0, 0, -1);
		Vector lookAt = Vector(cameraConfig.lookX, cameraConfig.lookY, cameraConfig.lookZ);

		float verticalFov = 20.f / 180 * 3.1415;
		viewHeight = 2.0 * tan(verticalFov);
		viewWidth = viewHeight * displayWidth / displayHeight;


		Vector up = Vector(0, 1, 0);

		Vector dir = (cameraPos - lookAt).getNormalizedVector();
		Vector w = up.crossProduct(dir);
		Vector h = dir.crossProduct(w);

		lowerRightPixelPos = w * viewWidth;
		upperLeftPixelPos = h * viewHeight;

		lowerLeftPixelPos = cameraPos - lowerRightPixelPos / 2 - upperLeftPixelPos / 2 - dir;
	}

	void writeImageToFile(string outputImage) {
		ofstream out("output_image.ppm");
		out << outputImage;
		out.close();
	}

	__device__ RgbPixel getBackgroundPixel(Ray ray) {
		return RgbPixel(0.8, 0.8, 1);
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

			RgbPixel color = Raytracer::getBackgroundPixel(ray);

			for (int i = 0; i < MAX_BOUNCES_PER_RAY; i++) {
				RayHitRecord hitRecord = Raytracer::getClosestRayHit(ray, objects, objectsCount);
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
		}
	}

	__global__ void createDemoWorld(Object** sceneObjects, ObjectConfig* objectConfigs, int objectsCount) {
		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i == objectsCount - 1) {
			sceneObjects[i] = new Sphere(Vector(0, -1000, 0), 1000, new Lambertian(RgbPixel(0.4, 0.7, 0.4)));
			return;
		} 

		if (i == objectsCount - 2) {
			sceneObjects[i] = new Sphere(Vector(5, 1, 3), 1, new Glass(RgbPixel(1.0, 1.0, 1.0), 1.33));
			return;
		}

		if (i == objectsCount - 3) {
			sceneObjects[i] = new Sphere(Vector(5, 1, 6), 1, new Metal(RgbPixel(0.8, 0.65, 0.65), 0.0));
			return;
		}

		if (i >= objectsCount - 5) {
			if (i == objectsCount - 4) {
				sceneObjects[i] = new Sphere(Vector(2, 1, 5), 0.95, new Glass(RgbPixel(1.0, 1.0, 1.0), 0.67));
				return;
			}
			sceneObjects[i] = new Sphere(Vector(2, 1, 5), 1, new Glass(RgbPixel(1.0, 1.0, 1.0), 1.5));
			return;
		}

		if (i >= objectsCount) return;

		int x = (i - objectsCount/2) / 10 + 10;
		int z = (i - objectsCount/2) % 10 + 10;

		curandState localRandState;
		curand_init(123, i, 0, &localRandState);

		float materialIndex = randomFloat(&localRandState);
		Vector center(x + 0.5 * randomFloat(&localRandState), 0.15, z + 0.5 * randomFloat(&localRandState));

		if ((center + Vector(-4, -0.2, 0)).getLength() > 0.9) {
			Material* material;

			if (materialIndex < 0.6) {
				material = new Lambertian(RgbPixel(randomFloat(&localRandState), randomFloat(&localRandState), randomFloat(&localRandState)));
				sceneObjects[i] = new Sphere(center, 0.2, material);
			}
			else if (materialIndex < 0.8) {
				material = new Metal(RgbPixel(0.0, 0.5, 0.5), 0.2);
				sceneObjects[i] = new Sphere(center, 0.2, material);
			}
			else {
				material = new Glass(RgbPixel(1.0, 1.0, 1.0), 1.33);
				sceneObjects[i] = new Sphere(center, 0.2, material);
			}
		}	
	}

	__global__ void freeWorld(Object** sceneObjects, int objectsCount) {
		for (int i = 0; i < objectsCount; i++) {
			delete* (sceneObjects + i);
		}
	}

	void renderImage(int setup) {

		// create scene objects on device
		Object** sceneObjects;
		ObjectConfig* devObjectConfigs;

		// Render scene based on config
		if (setup != 0) {
			cudaMalloc((void**)&sceneObjects, objectsCount * sizeof(Object*));
			cudaMalloc((void**)&devObjectConfigs, objectsCount * sizeof(ObjectConfig));
			cudaMemcpy(devObjectConfigs, objectConfigs, objectsCount * sizeof(ObjectConfig), cudaMemcpyHostToDevice);

			createWorld << <1, objectsCount >> > (sceneObjects, devObjectConfigs, objectsCount);
		}
		// Render demo scene
		else {
			objectsCount = 400;

			cudaMalloc((void**)&sceneObjects, objectsCount * sizeof(Object*));
			cudaMalloc((void**)&devObjectConfigs, objectsCount * sizeof(ObjectConfig));
			cudaMemcpy(devObjectConfigs, objectConfigs, objectsCount * sizeof(ObjectConfig), cudaMemcpyHostToDevice);

			createDemoWorld << <1, objectsCount >> > (sceneObjects, devObjectConfigs, objectsCount);
		}

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
}
