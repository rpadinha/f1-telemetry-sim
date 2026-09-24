#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

enum class TyreCompound {
    C1 = 0,
    C2 = 1,
    C3 = 2,
    C4 = 3,
    C5 = 4
};

enum class DriverAction {
    ACCELERATE,
    BRAKE,
    COAST
};

// struct for tyre properties
struct TyreProperties {
    float base_grip;                    // mu coeff (1.40 to 1.75)
    float opt_temp_c;                   // optimal working temperature
    float temp_window_c;                // window amplitude
    float thermal_deg_rate;             // overheating sensibility
    float wear_rate;                    // for each meter the ammount of wear
};

// the data that goes into the gpu
struct CarSetup {
    int id;                             // setup id
    float mass_kg;                      // Mass of the car
    float ice_power_kw;                 // Internal Combustion Engine Power
    float mguk_power_kw;                // Eletric Engine Power
    float drag_coef;                    // Aerodynamic Coefficient
};

// Keeps the state of the car every dt
struct F1Car {
    float a;                            // Linear acceleration
    float v;                            // Velocity of the car
    
    int current_seg;                    // Position of the car (segment of the circuit)
    float current_m;                    // Current meter of the circuit
    float time_s;                       // Time on track (resets after going through finish line)

    DriverAction action;                 // Current driver action

    float throttle_pedal;                // Throttle pedal position (0.0 to 1.0)
    float brake_pedal;                   // Brake pedal position (0.0 to 1.0)

    // Transmission
    int current_gear;                   // Current gear of the car
    float rpm;                          // Current RPM of the car
    float gear_shift_timer;             // ignition cut timer for upshifts (seconds)

    TyreCompound current_compound;      // Current tyre of the car (C1-hardest - C5-softest)
    float tyre_temp_front_c;            // average temp of front tyres
    float tyre_temp_rear_c;             // average temp of rear tyres
    float tyre_wear_pct;                // 0.0 (new) to 1.0 (worn out)

    float battery_mj;                   // Ammount of battery
    float fuel_kg;                      // Ammount of fuel

    bool drs_open;                      // drs boolean

    bool qualifying_mode;               // this is just cool to add for now but it states if its in qualifying mode or not, if it is then the car will not regenerate energy and will use more power to simulate a qualifying lap

    int laps_completed;                 // amount of laps completed by the sim for controlling better
};

// keeps the dynamics of the car each dt
struct F1CarDynamics {
    float max_grip;                     // 
    float long_grip;                    // 
    float drag_force;                   // 
    float lateral_force;                // 
    float gravity_longitudinal;         // 
    float max_traction_force;           // 
    float engine_breaking_force;        //
    float desired_braking_force;        //
    float desired_engine_force;         //
};

// track segment
struct TrackSegment {
    float length_m;                     // Length of the segment
    float radius_m;                     // Radius of the segment (< 10000 means curvature)
    float x;                            // x of segment
    float y;                            // y of segment
    float z;                            // z of segment
    bool drs_zone;                      // is the track segment a drs zone?
    float real_speed_kmh;               // Real speed of the segment (track data)
    float real_rpm;                     // Real RPM of the segment (track data)
    int real_gear;                      // Real gear of the segment (track data)
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

// this was created to export data to csv better
struct TelemetryPoint {
    float speed_kmh;                    // Speed in segment in km/h
    float rpm;                          // RPM in segment
    int gear;                           // Gear in segment
    float throttle;
    float brake;
    float time_s;
    float tyre_temp_front;
    float tyre_temp_rear;
};

CUDA_CALLABLE void step_physics(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt);

// This will start the kernel
void run_simulation_batch(const CarSetup* setups, SimResult* results, const TrackSegment* track, int num_segments, int numSetups);
void export_simulated_telemetry(const CarSetup& best_setup, const TrackSegment* d_track, int num_segments);

#endif