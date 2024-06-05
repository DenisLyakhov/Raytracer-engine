#pragma once
#include "Material.h"
#include "RgbPixel.h"

class Lambertian : public Material {
public:
	__device__ __host__ Lambertian() {}

	__device__ __host__ Lambertian(RgbPixel color) : Material(color) {}

	// Diffuse reflections
	__device__ Ray scatter(Ray ray, Vector normal, Vector hitPoint, bool isFront, curandState* randState) {
		Vector random = randomVector(normal, randState);
		if ((random + normal).isZero()) {
			return Ray(hitPoint, normal);
		}
		return Ray(hitPoint, random + normal);
	}
};
