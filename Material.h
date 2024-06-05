#pragma once
#include "Vector.h"
#include "Ray.h"

class Material {
protected:
	// Generates random vectors for diffuse lighting
	__device__ Vector randomVector(Vector normal, curandState* randState) {
		Vector randomDir = Vector(1, 1, 1);

		while (true) {
			randomDir = Vector(randomFloat(-1, 1, randState), randomFloat(-1, 1, randState), randomFloat(-1, 1, randState));

			if (randomDir.scalarProduct(randomDir) >= 1)
				continue;

			return randomDir.getNormalizedVector();
		}
	}

public:
	RgbPixel color;

	__device__ __host__ Material() {};

	__device__ __host__ Material(RgbPixel color) {
		this->color = color;
	}

	__device__ virtual Ray scatter(Ray ray, Vector normal, Vector hitPoint, bool isFront, curandState* randState) {
		return ray;
	}
};