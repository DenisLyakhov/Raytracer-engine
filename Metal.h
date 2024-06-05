#pragma once
#include "Material.h"
#include "RgbPixel.h"

class Metal : public Material {
public:
	float fuzziness = 0.0;

	__device__ __host__ Metal() {}

	__device__ __host__ Metal(RgbPixel color, float fuzziness) : Material(color) {
		this->fuzziness = fuzziness;
	}

	__device__ Ray scatter(Ray ray, Vector normal, Vector hitPoint, bool isFront, curandState* randState) {
		Ray scatteredRay = ray.reflect(normal, hitPoint);
		if (fuzziness > 0.0) {
			scatteredRay.vector = scatteredRay.vector + randomVector(normal, randState) * fuzziness;
		}
		return scatteredRay;
	}
};