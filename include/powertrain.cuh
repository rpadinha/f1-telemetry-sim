#ifndef POWERTRAIN_CUH
#define POWERTRAIN_CUH

#include "config.cuh"
#include "physics.cuh"

__host__ __device__ void update_transmission(F1Car* car) {
    float wheel_omega = car->v / Config::WHEEL_RADIUS;
    car->rpm = wheel_omega * Config::get_gear_ratio(car->current_gear) * Config::FINAL_DRIVE * 9.5492f; // 9.5492 is the conversion factor from rad/s to RPM

    if (car->rpm > Config::RPM_UPSHIFT && car->current_gear < 8) {
        car->current_gear++;
        car->rpm = wheel_omega * Config::get_gear_ratio(car->current_gear) * Config::FINAL_DRIVE * 9.5492f;
    } else if (car->rpm < Config::RPM_DOWNSHIFT && car->current_gear > 1) {
        car->current_gear--;
        car->rpm = wheel_omega * Config::get_gear_ratio(car->current_gear) * Config::FINAL_DRIVE * 9.5492f;
    }
}

#endif