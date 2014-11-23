/*
 * config.h
 *
 *  Created on: Nov 23, 2014
 *      Author: shiqing
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include <vector>
#include <string>
#include <ostream>

class Config
{
public:
	Config() { SetDefaults(); }
	~Config();
	Config(const std::string& path);

	bool quiet_mode;
	bool debug_mode;

	std::string	sequence_base_path;
	std::string	sequence_name;
	std::string	image_prefix;
	std::string	results_path;
	std::string	image_dir;

	int frame_width;
	int	frame_height;

	friend std::ostream& operator<< (std::ostream& out, const Config& conf);
private:
	void SetDefaults();
};

#endif /* CONFIG_H_ */
