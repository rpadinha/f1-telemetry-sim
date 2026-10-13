#ifndef AERO_CUH
#define AERO_CUH

#include "config.cuh"
#include "physics.cuh"

__host__ __device__ inline void calculate_aerodynamics(F1CarDynamics* dynamics, float car_speed, const CarSetup* setup, bool drs_open) {
    float current_drag_coef = setup->drag_coef;
    if (drs_open) { current_drag_coef *= 0.65f; }

    dynamics->drag_force = 0.5f * Config::AIR_DENSITY * (car_speed * car_speed) * current_drag_coef * Config::FRONTAL_AREA;
    // F_downforce = 0.5 * rho * v^2 * Cl * A 
    float cl_a = (setup->drag_coef * 4.5f) * Config::FRONTAL_AREA;
    dynamics->downforce = 0.5f * Config::AIR_DENSITY * (car_speed * car_speed) * cl_a;
}

#endif