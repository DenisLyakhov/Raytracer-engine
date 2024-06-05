#pragma once
#include "Material.h"
#include "RgbPixel.h"

class Glass : public Material {
private:
    float refractiveIndex = 0.5;

public:
    __device__ __host__ Glass() {}

    __device__ __host__ Glass(RgbPixel color, float refractiveIndex) : Material(color) {
        this->refractiveIndex = refractiveIndex;
    }

    __device__ Ray scatter(Ray ray, Vector normal, Vector hitPoint, bool isFront, curandState *randState) {
        float refractionRatio = refractiveIndex;

        if (isFront) {
            refractionRatio = 1.0 / refractionRatio;
        }

        Ray refractedRay = ray.refract(normal, hitPoint, refractionRatio, randState);

        return refractedRay;
    }
};
