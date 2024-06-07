#pragma once
#include "Object.h"
#include "math.h"

using namespace std;

class Cube : public Object {
public:
    Vector center;
    float radius;

    __device__ __host__ Cube(
        Vector center,
        float radius,
        Material* material
    ) : Object(material) {
        this->center = center;
        this->radius = radius;
    }

    __device__ RayHitRecord hitObject(Ray ray) {
        //Vector tmp = ray.origin - this->center;

        //float tMin = ((-tmp - this->radius) / ray.vector).getMaxValue();
        //float tMax = ((-tmp + this->radius) / ray.vector).getMinValue();

        //float t = tMin;

        //if (tMin > tMax) {
        //    return RayHitRecord();
        //}

        ///*if (t < MIN_T) {
        //    t = tMax;
        //    if (t < MIN_T) {
        //        return RayHitRecord();
        //    }
        //}*/

        //Vector point = ray.getLocation(t);

        //// d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}
        //float distance = sqrt((point.x - this->center.x)* (point.x - this->center.x) + (point.y - this->center.y)*(point.y - this->center.y) + (point.z - this->center.z)* (point.z - this->center.z));

        //Vector normal = (point - this->center) / distance;

        //return RayHitRecord(point, normal, t, this->material, ray.vector);

        float x0 = this->center.x + this->radius;
        float x1 = this->center.x - this->radius;

        float y0 = this->center.y - this->radius;
        float y1 = this->center.y + this->radius;

        float k = this->center.z + this->radius;

        float t = (k - ray.origin.z) / ray.vector.z;

        if (t < MIN_T || t > INFINITY) {
            return RayHitRecord();
        }

        float x = ray.origin.x + t * ray.vector.x;
        float y = ray.origin.y + t * ray.vector.y;
        if (x > x0 || x < x1 || y < y0 || y > y1) {
            return RayHitRecord();
        }

        return RayHitRecord(ray.getLocation(t), Vector(0, 0, 1), t, this->material, ray.vector);

    }
};
