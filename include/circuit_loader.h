#ifndef CIRCUIT_LOADER_H
#define CIRCUIT_LOADER_H

#include <vector>
#include <string>
#include "physics.cuh"

// circuit loader from csv file
std::vector<TrackSegment> load_circuit_csv(const std::string& filename);

#endif