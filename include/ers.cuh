#ifndef ERS_CUH
#define ERS_CUH

#include "config.cuh"
#include "physics.cuh"

__host__ __device__ inline float calculate_mguk_deployment(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
    if (car->action != DriverAction::ACCELERATE || car->v < 16.6f || car->battery_mj <= 0.f || car->current_gear <= 3) {
        return 0.0f;
    }

    float upcoming_straight_m = track[car->current_seg].length_m - car->current_m;
    int lookahead = (car->current_seg + 1) % num_segments;
    
    while (is_straight(track[lookahead].radius_m) && upcoming_straight_m < 2500.0f) {
        upcoming_straight_m += track[lookahead].length_m;
        lookahead = (lookahead + 1) % num_segments;
    }

    float ratio = 0.0f;
    if (upcoming_straight_m > 800.f) {
        ratio = 1.0f;
    } else if (upcoming_straight_m > 400.f) {
        ratio = 0.5f;
    } else {
        ratio = 0.2f;
    }

    float battery_ratio = car->battery_mj / Config::MAX_BATTERY_MJ;
    ratio *= battery_ratio;

    if (car->v*3.6f > 320.0f) {
        ratio *= 0.7f;
    }
    if (car->v*3.6f > 340.0f) {
        ratio *= 0.2f;
    }
    return ratio;
}

__host__ __device__ inline void update_ers(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt) {
    if (car->throttle_pedal > 0.0f) {
        float mguk_ratio = calculate_mguk_deployment(car, setup, track, num_segments);
        float mguk_power = (mguk_ratio > 0.0f) ? (setup->mguk_power_kw * mguk_ratio) : 0.0f;
        
        car->battery_mj -= (mguk_power * dt) / 1000.0f;
        if (car->battery_mj < 0.0f) car->battery_mj = 0.0f;
    }

    if (car->brake_pedal > 0.0f) {
        car->battery_mj += (Config::MGUK_REGEN_KW * car->brake_pedal * dt) / 1000.0f;
    } 
    else if (car->throttle_pedal == 0.0f) {
        car->battery_mj += (Config::MGUK_REGEN_KW * 0.25f * dt) / 1000.0f;
    }

    if (car->battery_mj > Config::MAX_BATTERY_MJ) {
        car->battery_mj = Config::MAX_BATTERY_MJ;
    }
}
#endif