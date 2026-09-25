#include <fstream>                  // for export_sim_telemetry
#include <string>                   // for file name thing
#include <iostream>                 // for export_sim_telemetry
#include "physics.cuh"              // gives acess to structs and functions made in cuda
#include "config.cuh"               // gives acess to global config variables
#include "engine.cuh"               // gives acess to engine functions
#include "powertrain.cuh"           // gives acess to powertrain functions
#include "tyres.cuh"                // gives acess to tyres functions
#include "math_utils.cuh"           // Gives acess to math functions needed
#include "sim_driver.cuh"           // gives acess to sim driver

__host__ __device__ void step_physics(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt) {

    float speed = get_allowed_speed(car, setup, track, num_segments);

    // just brake or accelerate
    // coasting is still meh
    if (car->v > speed + 0.2f) {
        car->action = DriverAction::BRAKE;
    } else {
        car->action = DriverAction::ACCELERATE;
    }

    
    if (!car->qualifying_mode && car->v > speed - 3.0f && car->v <= speed + 0.2f) {
        car->action = DriverAction::COAST;
    }
    
    
    car->drs_open = (track[car->current_seg].drs_zone && car->action == DriverAction::ACCELERATE);

    // creating a new car dynamics everytime is weird
    F1CarDynamics dynamics = calculate_car_dynamics(car, setup, track, num_segments);

    update_driver_pedals(car, dynamics, setup, track, num_segments, dt);
    update_ers(car, setup, track, num_segments, dt);
    burn_fuel(car, car->throttle_pedal, dt);

    float net_force = compute_net_force(car, dynamics);

    float a = net_force / dynamics.total_mass;  // nice using dynamics
    car->a = a;                                 // accel
    car->v += a * dt;                           // vel
    if (car->v < 0.0f) car->v = 0.0f;           // Prevent reverse tracking bugs
    car->current_m += car->v * dt;
    car->time_s += dt;

    while (car->current_m >= track[car->current_seg].length_m) {
        car->current_m -= track[car->current_seg].length_m;
        car->current_seg++;
        if (car->current_seg >= num_segments) {
            car->current_seg = 0;
            car->laps_completed++;
        }
    }

    upshift_cut(car,dt);
    update_transmission(car);
    update_tyres(car, dynamics, setup, dt);
}

__global__ void simulate_lap(const CarSetup* setups, SimResult* results, int num_setups, const TrackSegment* track, int num_segments) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_setups) {
        CarSetup setup = setups[idx];
        F1Car car;

        // Starter states for each setup
        car.v = track[0].real_speed_kmh / 3.6f;
        car.a = 0.0f;
        car.battery_mj = 4.0f;
        car.fuel_kg = 20.0f;
        car.rpm = track[0].real_rpm;
        car.current_gear = track[0].real_gear;
        car.gear_shift_timer = 0.0f;
        car.current_compound = TyreCompound::C3;
        car.tyre_temp_fl = 85.0f;
        car.tyre_temp_fr = 85.0f;
        car.tyre_temp_rl = 85.0f;
        car.tyre_temp_rr = 85.0f;
        car.tyre_wear_fl = 0.0f;
        car.tyre_wear_fr = 0.0f;
        car.tyre_wear_rl = 0.0f;
        car.tyre_wear_rr = 0.0f;
        car.current_seg = 0;
        car.time_s = 0.0f;
        car.qualifying_mode = false;
        car.laps_completed = 0;
        car.throttle_pedal = track[0].real_throttle_pedal;
        car.brake_pedal = track[0].real_brake_pedal;
        
        float t = 0.0f;
        float dt = 0.002f;
        float max_speed = 0.0f;
        
        while (car.laps_completed < 1 && car.time_s < 300.f) {
            step_physics(&car, &setup, track, num_segments, dt);
            
            if (car.v > max_speed) max_speed = car.v;
            t += dt;
        }
        
        results[idx].setup_id = setup.id;
        results[idx].lap_time = car.time_s;
        results[idx].top_speed_kmh = max_speed * 3.6f; 
        results[idx].battery_used_mj = car.battery_mj;
    }
}

__global__ void record_pole_telemetry(CarSetup setup, const TrackSegment* track, int num_segments, TelemetryPoint* out_telemetry) {
    F1Car car;

    car.v = track[0].real_speed_kmh / 3.6f;
    car.a = 0.0f;
    car.battery_mj = 4.0f;
    car.fuel_kg = 6.0f;
    car.rpm = track[0].real_rpm;
    car.current_gear = track[0].real_gear;
    car.gear_shift_timer = 0.0f;
    car.current_compound = TyreCompound::C3;
    car.tyre_temp_fl = 85.0f;
    car.tyre_temp_fr = 85.0f;
    car.tyre_temp_rl = 85.0f;
    car.tyre_temp_rr = 85.0f;
    car.tyre_wear_fl = 0.0f;
    car.tyre_wear_fr = 0.0f;
    car.tyre_wear_rl = 0.0f;
    car.tyre_wear_rr = 0.0f;
    car.current_seg = 0;
    car.current_m = 0.0f;
    car.time_s = 0.0f;
    car.qualifying_mode = false;
    car.laps_completed = 0;
    car.throttle_pedal = track[0].real_throttle_pedal;
    car.brake_pedal = track[0].real_brake_pedal;

    float dt = 0.002f;
    int last_recorded_seg = -1;

    while (car.laps_completed < 1 && car.time_s < 300.0f) {
        if (car.current_seg != last_recorded_seg && car.current_seg < num_segments) {
            out_telemetry[car.current_seg].speed_kmh = car.v * 3.6f;
            out_telemetry[car.current_seg].rpm = car.rpm;
            out_telemetry[car.current_seg].gear = car.current_gear;
            out_telemetry[car.current_seg].throttle = car.throttle_pedal;
            out_telemetry[car.current_seg].brake = car.brake_pedal;
            out_telemetry[car.current_seg].time_s = car.time_s;
            out_telemetry[car.current_seg].tyre_temp_front = (car.tyre_temp_fl + car.tyre_temp_fr) / 2.f;
            out_telemetry[car.current_seg].tyre_temp_rear = (car.tyre_temp_rl + car.tyre_temp_rr) / 2.f;
            last_recorded_seg = car.current_seg;
        }
        step_physics(&car, &setup, track, num_segments, dt);
    }
}

void run_simulation_batch(const CarSetup* setups, SimResult* results, const TrackSegment* track, int num_segments, int num_setups) {
    CarSetup* d_setups;
    TrackSegment* d_track;
    SimResult* d_results;
    
    size_t setups_size = num_setups * sizeof(CarSetup);
    size_t track_size = num_segments * sizeof(TrackSegment);
    size_t results_size = num_setups * sizeof(SimResult);

    cudaMalloc(&d_setups, setups_size);
    cudaMalloc(&d_track, track_size);
    cudaMalloc(&d_results, results_size);

    cudaMemcpy(d_setups, setups, setups_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_track, track, track_size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (num_setups + threads_per_block - 1) / threads_per_block;

    simulate_lap<<<blocks_per_grid, threads_per_block>>>(d_setups, d_results, num_setups, d_track, num_segments);

    cudaDeviceSynchronize();

    cudaMemcpy(results, d_results, results_size, cudaMemcpyDeviceToHost);

    cudaFree(d_setups);
    cudaFree(d_track);
    cudaFree(d_results);
}

void export_simulated_telemetry(const CarSetup& best_setup, const TrackSegment* h_track, int num_segments, std::string year, std::string gp, std::string session) {
    TrackSegment* d_track;
    TelemetryPoint* d_telemetry;

    size_t track_size = num_segments * sizeof(TrackSegment);
    size_t telemetry_size = num_segments * sizeof(TelemetryPoint);

    // Aloca pista e buffer de telemetria na GPU
    cudaMalloc(&d_track, track_size);
    cudaMalloc(&d_telemetry, telemetry_size);

    // Copia a pista da RAM (CPU) para a VRAM (GPU)
    cudaMemcpy(d_track, h_track, track_size, cudaMemcpyHostToDevice);

    record_pole_telemetry<<<1, 1>>>(best_setup, d_track, num_segments, d_telemetry);
    cudaDeviceSynchronize();

    TelemetryPoint* h_telemetry = new TelemetryPoint[num_segments];
    cudaMemcpy(h_telemetry, d_telemetry, telemetry_size, cudaMemcpyDeviceToHost);

    std::string export_path = "../data/" + year + "_" + gp + "_" + session + "_sim.csv";
    std::ofstream file(export_path);
    if (!file.is_open()) {
        std::cerr << "[CSV] Error opening" + export_path + "!\n";
        delete[] h_telemetry;
        cudaFree(d_track);
        cudaFree(d_telemetry);
        return;
    }

    file << "Distance_m,Sim_Speed,Sim_RPM,Sim_Gear,Sim_Throttle,Sim_Brake,Sim_Time_s,Tyre_Temp_Front_C,Tyre_Temp_Rear_C\n";

    for (int i = 0; i < num_segments; ++i) {
        file << i << ","
             << h_telemetry[i].speed_kmh << ","
             << h_telemetry[i].rpm << ","
             << h_telemetry[i].gear << ","
             << h_telemetry[i].throttle << ","
             << h_telemetry[i].brake << ","
             << h_telemetry[i].time_s << ","
             << h_telemetry[i].tyre_temp_front << ","
             << h_telemetry[i].tyre_temp_rear << "\n";
    }

    file.close();
    std::cout << "[GPU/CPU] Telemetry Exported to: " + export_path + "\n";

    delete[] h_telemetry;
    cudaFree(d_track);
    cudaFree(d_telemetry);
}