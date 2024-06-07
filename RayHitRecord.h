#pragma once
#include "Vector.h"
#include "Material.h"

#define MIN_T 0.001

struct RayHitRecord {
	bool hit = false;
	Vector point;
	Vector normal;
	float t = -1;
	Material* material;
	bool isFront = true;

	__device__ RayHitRecord() {}

	__device__ RayHitRecord(Vector point, Vector normal, float t, Material* material, Vector incomingVector) {
		this->hit = true;
		this->point = point;
		this->normal = normal;
		this->t = t;
		this->material = material;

		this->isFront = incomingVector.scalarProduct(normal) < 0.0;

		if (!isFront)
			this->normal = -normal;
	}

	__device__ Ray scatter(Ray incomingRay, curandState* randState) {
		return this->material->scatter(incomingRay, this->normal, this->point, this->isFront, randState);
	}
};
