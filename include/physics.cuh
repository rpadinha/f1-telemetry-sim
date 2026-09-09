#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

// the data that goes into the gpu
struct CarSetup {
    int id;                             // setup id
    float mass_kg;                      // Mass of the car
    float ice_power_kw;                 // Internal Combustion Engine Power
    float mguk_power_kw;                // Eletric Engine Power
    float drag_coef;                    // Aerodynamic Coefficient
};

enum class DriverAction {
    ACCELERATE,
    BRAKE,
    COAST
};

// Keeps the state of the car every dt
struct F1Car {
    float v;                            // Velocity of the car
    int current_seg;                    // Position of the car (segment of the circuit)
    float current_m;                    // Current meter of the circuit
    float battery_mj;                   // Ammount of battery
    float time_s;                       // Time on track (resets after going through finish line)

    DriverAction action;                 // Current driver action

    float throttle_pedal;                // Throttle pedal position (0.0 to 1.0)
    float brake_pedal;                   // Brake pedal position (0.0 to 1.0)

    bool qualifying_mode;               // this is just cool to add for now but it states if its in qualifying mode or not, if it is then the car will not regenerate energy and will use more power to simulate a qualifying lap

    // Transmission
    int current_gear;                   // Current gear of the car
    float rpm;                          // Current RPM of the car
};

// track segment
struct TrackSegment {
    float length_m;                     // Length of the segment
    float radius_m;                     // Radius of the segment (< 10000 means curvature)
    float x;                            // x of segment
    float y;                            // y of segment
    float real_speed_kmh;               // Real speed of the segment (track data)
    float real_rpm;                     // Real RPM of the segment (track data)
    int real_gear;                    // Real gear of the segment (track data)
    float real_throttle_pedal;          // Real value of throttle pedal
    float real_brake_pedal;             // Real value of brake pedal
};

// Results
struct SimResult {
    int setup_id;                       // Setup Id
    float lap_time;                     // Lap Time
    float top_speed_kmh;                // Top Speed
    float battery_used_mj;              // Ammount of Battery Used in MJ (starting + what is regenerated)
};

CUDA_CALLABLE void step_physics(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt);

// This will start the kernel
void run_simulation_batch(const CarSetup* setups, SimResult* results, const TrackSegment* track, int num_segments, int numSetups);

#endif