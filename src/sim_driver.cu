#include "sim_driver.cuh"
#include "math_utils.cuh"
#include <math.h>

__host__ __device__ float get_max_deceleration(float v_ms, float pitch_angle, const CarSetup* setup) {
    // first we calculate the aerodynamics at current speed v_ms
    float drag = 0.5f * Config::AIR_DENSITY * (v_ms * v_ms) * setup->drag_coef * Config::FRONTAL_AREA;
    float downforce = 0.5f * Config::AIR_DENSITY * (v_ms * v_ms) * (setup->drag_coef * 3.f) * Config::FRONTAL_AREA;

    // mechanical grip with curr tyres
    float normal_force = (setup->mass_kg * Config::GRAVITY * cosf(pitch_angle)) + downforce;
    if (normal_force < 0.0f) { normal_force = 0.0f; }
    float max_mech_brake = normal_force * Config::BASE_MECH_GRIP;

    // travagem total = brakes(tyres) + drag force
    float total_brake_force = max_mech_brake + drag;

    float base_decel = total_brake_force / setup->mass_kg;

    // gravity impacts if its going up / should also affect when its going down
    return base_decel + (Config::GRAVITY * sinf(pitch_angle));
}

__host__ __device__ float get_allowed_speed(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
    float current_pitch = get_track_pitch_angle(track, car->current_seg, num_segments);
    float downforce = 0.5f * Config::AIR_DENSITY * (car->v * car->v) * (setup->drag_coef * 3.f) * Config::FRONTAL_AREA;

    float current_normal = (setup->mass_kg * Config::GRAVITY * cosf(current_pitch)) + downforce;
    if (current_normal < 0.0f) current_normal = 0.0f;

    float max_grip = current_normal * Config::BASE_MECH_GRIP;

    float speed = sqrtf((max_grip * track[car->current_seg].radius_m) / setup->mass_kg);

    float dist_to_curve = track[car->current_seg].length_m - car->current_m;

    for (int i = 1; i <= Config::LOOKAHEAD_METERS; ++i) {
        int lookahead = (car->current_seg + i) % num_segments;
        
        if (!is_straight(track[lookahead].radius_m, setup)) {
            // including downforce logic at the corner will give us a more perfect approach the corner
            float future_pitch = get_track_pitch_angle(track, lookahead, num_segments);
            float future_normal = (setup->mass_kg * Config::GRAVITY * cosf(future_pitch)) + downforce;
            if (future_normal < 0.0f) future_normal = 0.0f;

            float future_grip = future_normal * Config::BASE_MECH_GRIP;
            float corner_v_sq = (future_grip * track[lookahead].radius_m) / setup->mass_kg;

            float corner_v = sqrtf(corner_v_sq);

            // Getting average G force in all braking zone
            float avg_speed_during_braking = (speed + corner_v) * 0.5f;
            float effective_decel = get_max_deceleration(avg_speed_during_braking, current_pitch, setup);
            // 20% cutoff for humanlike transition
            effective_decel *= 0.80f;
            if (effective_decel < 1.0f) { effective_decel = 1.0f; }

            float v_critical = sqrtf(corner_v_sq + (2.0f * effective_decel * dist_to_curve));
            
            if (v_critical < speed) {
                speed = v_critical;
            }
        }
        dist_to_curve += track[lookahead].length_m; 
    }
    return speed;
}

__host__ __device__ float calculate_mguk_deployment(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
    if (car->action != DriverAction::ACCELERATE || car->v < 16.6f || car->battery_mj <= 0.f || !is_straight(track[car->current_seg].radius_m, setup)) {
        return 0.0f;
    }

    float upcoming_straight_m = track[car->current_seg].length_m - car->current_m;
    int lookahead = (car->current_seg + 1) % num_segments;
    
    while (is_straight(track[lookahead].radius_m, setup) && upcoming_straight_m < 2500.0f) {
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

    if (car->v*3.6f > 320.0f) {
        ratio *= 0.7f;
    }
    if (car->v*3.6f > 340.0f) {
        ratio *= 0.2f;
    }
    return ratio;
}