#pragma once

#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "Sphere.h"

#define SPHERE 0
#define DEFAULT -1

#define LAMBERTIAN 0
#define GLASS 1
#define METAL 2

struct CameraConfig {
	float x;
	float y;
	float z;
	float lookX;
	float lookY;
	float lookZ;
};

struct ObjectConfig {
	int type;
	int material;
	float r;
	float g;
	float b;
	float x;
	float y;
	float z;
	float height;
	float radius = 0;
	float fuzziness = 0;
	float refractiveIndex = 0;
};

nlohmann::json loadedFile;

CameraConfig cameraConfig;

std::vector<Object*> objectsVector;
int objectsCount;

ObjectConfig* objectConfigs;

class JsonParser {
public:
	static void loadJsonObjectConfigs();
};

constexpr unsigned int str2int(const char* str, int h = 0) {
	return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

void JsonParser::loadJsonObjectConfigs() {
	std::ifstream inputFile("data.json");
	if (inputFile.is_open()) {
		inputFile >> loadedFile;
		inputFile.close();
	}

	auto object = loadedFile["config"][0];

	cameraConfig.x = (float)object["x"];
	cameraConfig.y = (float)object["y"];
	cameraConfig.z = (float)object["z"];

	cameraConfig.lookX = (float)object["lookX"];
	cameraConfig.lookY = (float)object["lookY"];
	cameraConfig.lookZ = (float)object["lookZ"];

	objectsCount = loadedFile["elements"].size();
	objectConfigs = new ObjectConfig[objectsCount];

	for (int index = 0; index < objectsCount; index++) {
		int i = index;

		auto object = loadedFile["elements"][i];

		std::string type = (std::string)object["type"].get<std::string>();

		switch (str2int(type.c_str())) {
		case str2int("sphere"):
			objectConfigs[i].type = SPHERE;
			objectConfigs[i].radius = (float)object["radius"];
			break;
		default:
			objectConfigs[i].type = DEFAULT;
			break;
		}

		objectConfigs[i].r = (float)object["material"]["rgbpixel"]["r"];
		objectConfigs[i].g = (float)object["material"]["rgbpixel"]["g"];
		objectConfigs[i].b = (float)object["material"]["rgbpixel"]["b"];

		objectConfigs[i].x = (float)object["vector"]["x"];
		objectConfigs[i].y = (float)object["vector"]["y"];
		objectConfigs[i].z = (float)object["vector"]["z"];

		std::string materialType = (std::string)object["material"]["type"].get<std::string>();

		switch (str2int(materialType.c_str())) {
		case str2int("glass"):
			objectConfigs[i].material = GLASS;
			objectConfigs[i].refractiveIndex = object["material"]["refractiveIndex"];
			break;
		case str2int("metal"):
			objectConfigs[i].material = METAL;
			objectConfigs[i].fuzziness = object["material"]["fuzziness"];
			break;
		case str2int("lambertian"):
			objectConfigs[i].material = LAMBERTIAN;
			break;
		default:
			objectConfigs[i].material = DEFAULT;
			break;
		}
	}
}