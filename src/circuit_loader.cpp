#include "circuit_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<TrackSegment> load_circuit_csv(const std::string& filename) {
    std::vector<TrackSegment> track;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "[ERROR] Not possible to open circuit file" << filename << std::endl;
        return track;
    }

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string length_str, radius_str, x_str, y_str, z_str, speed_str, rpm_str, gear_str, throttle_str, brake_str;

        if (std::getline(ss, length_str, ',') && 
                std::getline(ss, radius_str, ',') &&
                std::getline(ss, x_str, ',') &&
                std::getline(ss, y_str, ',') && 
                std::getline(ss, z_str, ',') &&
                std::getline(ss, speed_str, ',') &&
                std::getline(ss, rpm_str, ',') &&
                std::getline(ss, gear_str, ',') &&
                std::getline(ss, throttle_str, ',') &&
                std::getline(ss, brake_str, ',')
            ) {
            
            TrackSegment seg;
            seg.length_m = std::stof(length_str);
            seg.radius_m = std::stof(radius_str);
            seg.x = std::stof(x_str);
            seg.y = std::stof(y_str);
            seg.z = std::stof(z_str);
            seg.real_speed_kmh = std::stof(speed_str);
            seg.real_rpm = std::stof(rpm_str);
            seg.real_gear = std::stof(gear_str);
            seg.real_throttle_pedal = std::stof(throttle_str);
            seg.real_brake_pedal = std::stof(brake_str);
            track.push_back(seg);
        }
    }
    return track;
}