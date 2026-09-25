#include <iostream>                 // cout things
#include <cstdlib>                  // for argument handling
#include <string>                   // for argument handling
#include <vector>                   // for track segments, car setups and results
#include <random>                   // for random
#include <algorithm>                // for rnd

// fallback in case Cmake doesnt do it
#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR "."
#endif

#include "physics.cuh"              // cudaaaaaa
#include "config.cuh"               // configs - shared between cuda and here
#include "circuit_loader.h"         // circuit loader helper for csv handling
#include "visualizer.h"             // visualizer (SFML)

int main(int argc, char* argv[]) {

    if (argc < 4) {
        std::cout << "Correct usage: " << argv[0] << "<year> <track> <session>" << std::endl;
        std::cout << "example: " << argv[0] << "2023 Baku Q" << std::endl;
        return 1;
    }

    std::string year = argv[1];
    std::string gp = argv[2];
    std::string session = argv[3];
    std::string root = PROJECT_ROOT_DIR;

    std::string python_exe = root + "/.f1env/bin/python";
    std::string data_script_path = root + "/scripts/fetch_f1data.py";

    std::string import_command = python_exe + " " + data_script_path + " " + year + " " + gp + " " + session;
    int import_result = std::system(import_command.c_str());

    if (import_result != 0) {
        std::cerr << "Error executing import python script.\n";
        return 1;
    }

    const std::string IMPORT_FILE = root + "/data/" + year + "_" + gp + "_" + session + ".csv";
    const std::string EXPORT_FILE = root + "/data/" + year + "_" + gp + "_" + session + "_sim.csv";
    std::cout << "[CPU] Loading circuit from: " << IMPORT_FILE << std::endl;

    std::vector<TrackSegment> track = load_circuit_csv(IMPORT_FILE);

    if (track.empty()) return -1;
    
    std::cout << "[CPU] Track loaded! Total segments: " << track.size() << std::endl;
    
    std::cout << "[CPU] Generating " << Config::NUM_SETUPS << " different setups..." << std::endl;

    std::vector<CarSetup> setups(Config::NUM_SETUPS);
    std::vector<SimResult> results(Config::NUM_SETUPS);


    // after getting the battery and fixing some imperfections as to not taking the Z of the track, we will 
    // see different ways to change setups*
    std::mt19937 gen(888); 
    std::uniform_real_distribution<float> ice_dist(380.0f, 420.0f);
    std::uniform_real_distribution<float> mguk_dist(300.0f, 350.0f);
    std::uniform_real_distribution<float> drag_dist(0.7f, 1.5f);

    for(int i = 0 ; i < Config::NUM_SETUPS ; ++i) {
        setups[i].id = i;
        setups[i].mass_kg = 798.0f;
        setups[i].ice_power_kw = ice_dist(gen);
        setups[i].mguk_power_kw = mguk_dist(gen);
        setups[i].drag_coef = drag_dist(gen);
    }

    std::cout << "[CUDA] Sending Data to GPU..." << std::endl;

    run_simulation_batch(setups.data(), results.data(), track.data(), track.size(), Config::NUM_SETUPS);

    std::cout << "[CPU] Success here are the results: \n\n";

    std::sort(results.begin(), results.end(), [](const SimResult& a, const SimResult& b) {
        return a.lap_time < b.lap_time;
    });

    CarSetup pole_position_setup = setups[results[0].setup_id];

    std::cout << " ----- Pole Position -----" << std::endl;
    std::cout << " ----- Setup Id: " << results[0].setup_id << std::endl;
    std::cout << pole_position_setup.ice_power_kw << "kw(ICE)|" << pole_position_setup.mguk_power_kw << "kw(MGU-K)|" << 
    pole_position_setup.drag_coef << "(DRAG)" << std::endl;
    std::cout << " ----- Lap Time: " << results[0].lap_time << std::endl;
    std::cout << " ----- Max Velocity: " << results[0].top_speed_kmh << std::endl;

    export_simulated_telemetry(pole_position_setup, track.data(), track.size(), year, gp, session);

    run_sfml_visualizer(track, pole_position_setup);

    std::string plot_script_path = root + "/scripts/plot_comparisons.py";
    std::string export_command = python_exe + " " + plot_script_path + " " + IMPORT_FILE + " " + EXPORT_FILE;


    int export_result = std::system(export_command.c_str());
    if (export_result != 0) {
        std::cerr << "Error executing export python script.\n";
        return 1;
    }
}