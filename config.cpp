/*
 * config.cpp
 *
 *  Created on: Nov 23, 2014
 *      Author: shiqing
 */

#include "config.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
Config::Config(const std::string& path) {
	SetDefaults();

	ifstream f(path.c_str());
	if (!f) {
		cout << "error: could not load config file: " << path << endl;
		return;
	}

	string line, name, tmp;
	while (getline(f, line)) {
		istringstream iss(line);
		iss >> name >> tmp;

		// skip invalid lines and comments
		if (iss.fail() || tmp != "=" || name[0] == '#')
			continue;

		else if (name == "quietMode")
			iss >> quiet_mode;
		else if (name == "debugMode")
			iss >> debug_mode;
		else if (name == "sequenceBasePath")
			iss >> sequence_base_path;
		else if (name == "sequenceName")
			iss >> sequence_name;
		else if (name == "imageDir")
				iss >> image_dir;
		else if (name == "resultsPath")
			iss >> results_path;
		else if (name == "frameWidth")
			iss >> frame_width;
		else if (name == "frameHeight")
			iss >> frame_height;
		else if (name == "imagePrefix")
			iss >> image_prefix;

	}
}


void Config::SetDefaults()
{

	quiet_mode = false;
	debug_mode = false;

	sequence_base_path = "";
	sequence_name = "";
	results_path = "";

	frame_width = 320;
	frame_height = 240;
}


ostream& operator<< (ostream& out, const Config& conf)
{
	out << "config:" << endl;
	out << "  quietMode          = " << conf.quiet_mode << endl;
	out << "  debugMode          = " << conf.debug_mode << endl;
	out << "  sequenceBasePath   = " << conf.sequence_base_path << endl;
	out << "  sequenceName       = " << conf.sequence_name << endl;
	out << "  resultsPath        = " << conf.results_path << endl;
	out << "  frameWidth         = " << conf.frame_width << endl;
	out << "  frameHeight        = " << conf.frame_height << endl;

	return out;
}

Config::~Config() {
	// TODO Auto-generated destructor stub
}

