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

    float target_speed = get_allowed_speed(car, setup, track, num_segments);
    float speed_error = car->v - target_speed;

    if (speed_error > 0.3f) {
        car->action = DriverAction::BRAKE;
    } else if (car->action == DriverAction::BRAKE && speed_error > -2.0f) {
        car->action = DriverAction::COAST;
    } else {
        car->action = DriverAction::ACCELERATE;
    }

    car->drs_open = (track[car->current_seg].drs_zone && car->action == DriverAction::ACCELERATE);

    // 1. Compute base vehicle dynamics once per dt and store in car->dynamics
    car->dynamics = calculate_car_dynamics(car, setup, track, num_segments);

    // 2. Driver & Power Unit updates (populates car->dynamics.desired_engine_force)
    update_driver_pedals(car, setup, track, num_segments, dt);
    update_ers(car, setup, track, num_segments, dt);
    burn_fuel(car, car->throttle_pedal, dt);

    // 3. Net longitudinal force calculation
    float net_force = compute_net_force(car);
    car->dynamics.applied_long_force = net_force;

    // 4. Kinematic integration
    float new_accel = net_force / car->dynamics.total_mass;
    car->a = new_accel;
    car->v += new_accel * dt;
    if (car->v < 0.0f) car->v = 0.0f;
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

    // 5. Transmission & Tyre updates
    upshift_cut(car, dt);
    update_transmission(car);
    update_tyres(car, dt);
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
        car.fuel_kg = 6.0f;
        car.current_gear = track[0].real_gear;
        car.gear_shift_timer = 0.0f;
        car.current_compound = TyreCompound::C3;
        car.tyre_temp_fl = 95.0f;
        car.tyre_temp_fr = 95.0f;
        car.tyre_temp_rl = 95.0f;
        car.tyre_temp_rr = 95.0f;
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
    car.tyre_temp_fl = 95.0f;
    car.tyre_temp_fr = 95.0f;
    car.tyre_temp_rl = 95.0f;
    car.tyre_temp_rr = 95.0f;
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
        int seg_before_step = car.current_seg;

        // Advance physics first so car.dynamics has all complete forces for this step
        step_physics(&car, &setup, track, num_segments, dt);

        if (seg_before_step != last_recorded_seg && seg_before_step < num_segments) {
            TelemetryPoint& pt = out_telemetry[seg_before_step];
            const F1CarDynamics& dyn = car.dynamics;

            pt.speed_kmh = car.v * 3.6f;
            pt.rpm = car.rpm;
            pt.gear = car.current_gear;
            pt.throttle = car.throttle_pedal;
            pt.brake = car.brake_pedal;
            pt.time_s = car.time_s;
            pt.tyre_temp_front = (car.tyre_temp_fl + car.tyre_temp_fr) * 0.5f;
            pt.tyre_temp_rear  = (car.tyre_temp_rl + car.tyre_temp_rr) * 0.5f;

            pt.tyre_temp_fl = car.tyre_temp_fl;
            pt.tyre_temp_fr = car.tyre_temp_fr;
            pt.tyre_temp_rl = car.tyre_temp_rl;
            pt.tyre_temp_rr = car.tyre_temp_rr;

            pt.total_mass           = dyn.total_mass;
            pt.pitch_angle          = dyn.pitch_angle;
            pt.drag_force           = dyn.drag_force;
            pt.downforce            = dyn.downforce;
            pt.normal_force         = dyn.normal_force;
            pt.lateral_force        = dyn.lateral_force;
            pt.gravity_longitudinal = dyn.gravity_longitudinal;
            pt.max_grip             = dyn.max_grip;
            pt.long_grip            = dyn.long_grip;
            pt.max_traction_force   = dyn.max_traction_force;
            pt.engine_braking_force = get_engine_braking_force(car.rpm);
            pt.desired_engine_force = dyn.desired_engine_force;
            pt.applied_long_force   = dyn.applied_long_force;
            pt.load_fl              = dyn.load_fl;
            pt.load_fr              = dyn.load_fr;
            pt.load_rl              = dyn.load_rl;
            pt.load_rr              = dyn.load_rr;

            last_recorded_seg = seg_before_step;
        }
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

    cudaMalloc(&d_track, track_size);
    cudaMalloc(&d_telemetry, telemetry_size);

    cudaMemcpy(d_track, h_track, track_size, cudaMemcpyHostToDevice);

    record_pole_telemetry<<<1, 1>>>(best_setup, d_track, num_segments, d_telemetry);
    cudaDeviceSynchronize();

    TelemetryPoint* h_telemetry = new TelemetryPoint[num_segments];
    cudaMemcpy(h_telemetry, d_telemetry, telemetry_size, cudaMemcpyDeviceToHost);

    std::string export_path = "../data/" + year + "_" + gp + "_" + session + "_sim.csv";
    std::ofstream file(export_path);
    if (!file.is_open()) {
        std::cerr << "[CSV] Error opening " + export_path + "!\n";
        delete[] h_telemetry;
        cudaFree(d_track);
        cudaFree(d_telemetry);
        return;
    }

    file << "Distance_m,Sim_Speed,Sim_RPM,Sim_Gear,Sim_Throttle,Sim_Brake,Sim_Time_s,"
         << "Tyre_Temp_Front_C,Tyre_Temp_Rear_C,Tyre_Temp_FL_C,Tyre_Temp_FR_C,Tyre_Temp_RL_C,Tyre_Temp_RR_C,"
         << "Total_Mass_kg,Pitch_Angle_rad,Drag_Force_N,Downforce_N,Normal_Force_N,Lateral_Force_N,Gravity_Long_N,"
         << "Max_Grip_N,Long_Grip_N,Max_Traction_N,Engine_Braking_N,Desired_Engine_N,"
         << "Applied_Long_Force_N,Load_FL_N,Load_FR_N,Load_RL_N,Load_RR_N\n";

    for (int i = 0; i < num_segments; ++i) {
        const TelemetryPoint& pt = h_telemetry[i];
        file << i << ","
             << pt.speed_kmh << "," << pt.rpm << "," << pt.gear << ","
             << pt.throttle << "," << pt.brake << "," << pt.time_s << ","
             << pt.tyre_temp_front << "," << pt.tyre_temp_rear << ","
             << pt.tyre_temp_fl << "," << pt.tyre_temp_fr << ","
             << pt.tyre_temp_rl << "," << pt.tyre_temp_rr << ","
             << pt.total_mass << "," << pt.pitch_angle << ","
             << pt.drag_force << "," << pt.downforce << "," << pt.normal_force << ","
             << pt.lateral_force << "," << pt.gravity_longitudinal << ","
             << pt.max_grip << "," << pt.long_grip << ","
             << pt.max_traction_force << "," << pt.engine_braking_force << ","
             << pt.desired_engine_force << "," << pt.applied_long_force << ","
             << pt.load_fl << "," << pt.load_fr << "," << pt.load_rl << "," << pt.load_rr << "\n";
    }

    file.close();
    std::cout << "[GPU/CPU] Telemetry Exported to: " + export_path + "\n";

    delete[] h_telemetry;
    cudaFree(d_track);
    cudaFree(d_telemetry);
}