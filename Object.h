#pragma once
#include "Material.h"
#include "RayHitRecord.h"

class Object {
public:
	Material* material;

	__device__ Object(Material* material) { this->material = material; }

	__device__ virtual RayHitRecord hitObject(Ray ray) {
		return RayHitRecord();
	};
};