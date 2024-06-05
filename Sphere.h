#pragma once
#include "Object.h"

class Sphere : public Object {
public:
    Vector center;
    float radius;

    __device__ __host__ Sphere(
        Vector center,
        float radius,
        Material* material
    ) : Object(material) {
        this->center = center;
        this->radius = radius;
    }

    __device__ RayHitRecord hitObject(Ray ray) {
        Vector tmp = ray.origin - this->center;

        float a = ray.vector.scalarProduct(ray.vector);
        float b = 2.0 * tmp.scalarProduct(ray.vector);
        float c = tmp.scalarProduct(tmp) - this->radius * this->radius;

        float det = b * b - 4 * a * c;
        if (det < 0) return RayHitRecord();

        // Need to take into account both sides!
        float t = (-b - sqrt(det)) / (2. * a);
        if (t < MIN_T) { 
            t = (-b + sqrt(det)) / (2. * a);
            if (t < MIN_T)
                return RayHitRecord();
        }

        Vector point = ray.getLocation(t);
        Vector normal = (point - this->center) / this->radius;

        return RayHitRecord(point, normal, t, this->material, ray.vector);
    }
};
