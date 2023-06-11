#pragma once

#include "Common.cuh"
#include <glm/gtc/matrix_transform.hpp>
#include <curand.h>
#include <curand_kernel.h>

inline __device__ glm::vec3 random_in_unit_sphere(curandState* randState)
{
	glm::vec3 p;
	p = 2.0f * glm::vec3(curand_uniform(randState), curand_uniform(randState), curand_uniform(randState)) - glm::vec3(1.0f);
	return p;
}



class Ray
{
public:
	__device__ Ray()
	{
	}

	__device__ Ray(const glm::vec3& a, const glm::vec3& b)
	{
		A = a;
		B = b;
	}

	__device__ glm::vec3 origin() const { return A; }
	__device__ glm::vec3 direction() const { return B; }
	__device__ glm::vec3 point_at_parameter(float t) const { return A + t * B; }

	glm::vec3 A;
	glm::vec3 B;
};

class Material;

struct HitRecord
{
	float t;
	glm::vec3 p;
	glm::vec3 normal;
	Material* material;
};

class Hitable
{
public:
	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};;

class HitableList : public Hitable
{
public:
	__device__ HitableList()
	{
	}

	__device__ HitableList(Hitable** l, int n)
	{
		list = l;
		list_size = n;
	}

	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const
	{
		HitRecord temp_rec;
		bool hit_anything = false;
		double closest_so_far = tmax;
		for (int i = 0; i < list_size; i++)
		{
			if (list[i]->hit(r, tmin, closest_so_far, temp_rec))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}

	Hitable** list;
	int list_size;
};

class Sphere : public Hitable
{
public:
	__device__ Sphere()
	{
	}

	__device__ Sphere(glm::vec3 cen, float r, Material* m)
	{
		center = cen;
		radius = r;
		material = m;
	}

	__device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override
	{
		glm::vec3 oc = r.origin() - center;
		float a = glm::dot(r.direction(), r.direction());
		float b = glm::dot(oc, r.direction());
		float c = glm::dot(oc, oc) - radius * radius;
		float discriminant = b * b - a * c;
		if (discriminant > 0)
		{
			float temp = (-b - sqrt(discriminant)) / a;
			if (temp < t_max && temp > t_min)
			{
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.material = material;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp < t_max && temp > t_min)
			{
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.material = material;
				return true;
			}
		}
		return false;
	}

	glm::vec3 center;
	float radius;
	Material* material;
};


class Camera
{
public:
	glm::vec3 position;
	__device__ Ray GetRay(glm::vec2 m)
	{
		return Ray(position, {m.x, m.y, -1.0f});
	}
};

class Material
{
public:
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered,
	                                curandState* randState) const = 0;
};

class Lambertian : public Material
{
public:
	__device__ Lambertian(const glm::vec3& a)
	{
		albedo = a;
	}

	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered,
	                                curandState* randState) const override
	{
		glm::vec3 target = rec.p + rec.normal + random_in_unit_sphere(randState);
		scattered = Ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}

	glm::vec3 albedo;
};
class Metal: public Material
{
	public:
	__device__ Metal(const glm::vec3& a, float f)
	{
		albedo = a;
		fuzz = f;
	}
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered,
	                                curandState* randState) const override
	{
		glm::vec3 reflected = glm::reflect(glm::normalize(r_in.direction()), rec.normal);
		scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(randState));
		attenuation = albedo;
		return (glm::dot(scattered.direction(), rec.normal) > 0);
	}
	glm::vec3 albedo;
	float fuzz;
};;

void Render(glm::vec2 size, uint32_t* pixels, Camera* camera, curandState* randState);
void Init(int width, int height, curandState* randState);
