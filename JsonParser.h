#pragma once

#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "Sphere.h"
#include "Cube.h"

#define SPHERE 0
#define CONE 1
#define CUBE 2
#define DEFAULT -1

#define LAMBERTIAN 0
#define GLASS 1
#define METAL 2

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
std::vector<Object*> objectsVector;
int objectsCount;

ObjectConfig* objectConfigs;

class JsonParser {
private:
	//Creating objects declarations
	static Object* createObject(nlohmann::json sphere);
	static Sphere* createSphere(nlohmann::json sphere);
	static Cube* createCube(nlohmann::json cube);

	//Creating materials declaration
	static Material* createMaterial(nlohmann::json material);
	static Glass* createGlass(nlohmann::json glass);
	static Metal* createMetal(nlohmann::json metal);
	static Lambertian* createLambertian(nlohmann::json lambertian);

	//create helpers
	static Vector createVector(nlohmann::json vector);
	static RgbPixel createRgbPixel(nlohmann::json rgbpixel);
public:
	static void loadJsonFile();
	static void loadJsonObjectConfigs();
};

void JsonParser::loadJsonFile() {
	std::ifstream inputFile("data.json");
	if (inputFile.is_open()) {
		inputFile >> loadedFile;
		inputFile.close();
	}
	if (loadedFile["elements"].is_array()) {
		for (const auto& element : loadedFile["elements"]) {
			objectsVector.push_back(createObject(element));
			// createSphere(element)
		}
	}
}

constexpr unsigned int str2int(const char* str, int h = 0) {
	return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

void JsonParser::loadJsonObjectConfigs() {
	std::ifstream inputFile("data.json");
	if (inputFile.is_open()) {
		inputFile >> loadedFile;
		inputFile.close();
	}

	objectsCount = loadedFile["elements"].size();
	objectConfigs = new ObjectConfig[objectsCount];

	for (int i = 0; i < objectsCount; i++) {
		auto object = loadedFile["elements"][i];

		std::string type = (std::string)object["type"].get<std::string>();

		switch (str2int(type.c_str())) {
		case str2int("sphere"):
			objectConfigs[i].type = SPHERE;
			objectConfigs[i].radius = (float)object["radius"];
			break;
		case str2int("cone"):
			objectConfigs[i].type = CONE;
			objectConfigs[i].height = (float)object["height"];
			objectConfigs[i].radius = (float)object["radius"];
			break;
		case str2int("cube"):
			objectConfigs[i].type = CUBE;
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


Object* JsonParser::createObject(nlohmann::json object) {
	std::string value = (std::string)object["type"].get<std::string>();
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return std::tolower(c);
		});

	Object* returnObject;
	switch (str2int(value.c_str())) {
	case str2int("sphere"):
		returnObject = createSphere(object);
		break;
	case str2int("cube"):
		returnObject = createCube(object);
		break;
	default:
		returnObject = createSphere(object);
		break;
	}
	return returnObject;
}

Sphere* JsonParser::createSphere(nlohmann::json sphere) {
	return new Sphere(createVector(sphere["vector"]), (float)sphere["radius"], createMaterial(sphere["material"]));
}

Cube* JsonParser::createCube(nlohmann::json cube) {
	return new Cube(createVector(cube["vector"]), (float)cube["radius"], createMaterial(cube["material"]));
}

Material* JsonParser::createMaterial(nlohmann::json material) {
	std::string value = (std::string)material["type"].get<std::string>();
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return std::tolower(c);
		});

	Material* returnMaterial;
	switch (str2int(value.c_str())) {
	case str2int("glass"):
		returnMaterial = createGlass(material);
		break;
	case str2int("metal"):
		returnMaterial = createMetal(material);
		break;
	case str2int("lambertian"):
		returnMaterial = createLambertian(material);
		break;
	default:
		returnMaterial = material.contains("rgbpixel") ? new Material(createRgbPixel(material["rgbpixel"])) : new Material();
		break;
	}

	return returnMaterial;
}
Glass* JsonParser::createGlass(nlohmann::json glass) {
	return new Glass(createRgbPixel(glass["rgbpixel"]), (float)glass["refractiveIndex"]);
}
Metal* JsonParser::createMetal(nlohmann::json metal) {
	return new Metal(createRgbPixel(metal["rgbpixel"]), (float)metal["fuzziness"]);
}
Lambertian* JsonParser::createLambertian(nlohmann::json lambertian) {
	return new Lambertian(createRgbPixel(lambertian["rgbpixel"]));
}


RgbPixel JsonParser::createRgbPixel(nlohmann::json rgbpixel) {
	return RgbPixel((float)rgbpixel["r"], (float)rgbpixel["g"], (float)rgbpixel["b"]);
}

Vector JsonParser::createVector(nlohmann::json vector) {
	return Vector((float)vector["x"], (float)vector["y"], (float)vector["z"]);
}