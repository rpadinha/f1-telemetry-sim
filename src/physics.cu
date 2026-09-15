#include "physics.cuh"
#include "config.cuh"
#include "powertrain.cuh"           // gives acess to powertrain functions
#include "math_utils.cuh"           // Gives acess to math functions needed
#include "sim_driver.cuh"           // gives acess to sim driver

// crazy crazy
__host__ __device__ void step_physics(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt) {

    float speed = get_allowed_speed(car, setup, track, num_segments);
    // this is how we lift rn maybe we are bad at this?
    float lift_threshold_speed = speed - 10.0f;
    if (car->v > speed) {
        car->action = DriverAction::BRAKE;
    } else if (car->v > lift_threshold_speed && !car->qualifying_mode) {
        car->action = DriverAction::COAST;
    } else if (car->v < lift_threshold_speed){ 
        car->action = DriverAction::ACCELERATE;
    }

    car->drs_open = (track[car->current_seg].drs_zone && car->action == DriverAction::ACCELERATE);
    F1CarDynamics dynamics = calculate_car_dynamics(car, setup, track, num_segments);

    update_driver_pedals(car, dynamics, setup, track, num_segments, dt);

    float net_force = compute_net_force(car, setup, dynamics);
    
    float fuel_flow_rate = (100.0f / 3600.0f);          // FIA 100kg/h
    if (car->fuel_kg > 0.0f) {
        car->fuel_kg -= (car->throttle_pedal * fuel_flow_rate) * dt;
        if (car->fuel_kg < 0.0f) car->fuel_kg = 0.0f;
    }
    
    float total_mass = setup->mass_kg + car->fuel_kg;

    float a = net_force / total_mass;
    car->v += a * dt;
    if (car->v < 0.0f) car->v = 0.0f; // Prevent reverse tracking bugs
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
    update_transmission(car);
}

// Cuda Kernel
__global__ void simulate_lap(const CarSetup* setups, SimResult* results, int num_setups, const TrackSegment* track, int num_segments) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_setups) {
        CarSetup setup = setups[idx];
        F1Car car;

        // Starter states for each setup
        car.v = track[0].real_speed_kmh / 3.6f;
        car.battery_mj = 4.0f;
        car.fuel_kg = 10.0f;
        car.rpm = track[0].real_rpm;
        car.current_gear = track[0].real_gear;
        car.current_seg = 0;
        car.time_s = 0.0f;
        car.qualifying_mode = true;
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