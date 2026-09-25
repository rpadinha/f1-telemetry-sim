#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include <string>

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

// struct for setup data (goes into gpu)
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
    
    // time on track, and current m and current segment
    int current_seg;                    // Position of the car (segment of the circuit)
    float current_m;                    // Current meter of the circuit
    float time_s;                       // Time on track (resets after going through finish line)

    // action and pedals
    DriverAction action;                 // Current driver action
    float throttle_pedal;                // Throttle pedal position (0.0 to 1.0)
    float brake_pedal;                   // Brake pedal position (0.0 to 1.0)

    // Transmission
    int current_gear;                   // Current gear of the car
    float rpm;                          // Current RPM of the car
    float gear_shift_timer;             // ignition cut timer for upshifts (seconds)

    // Tyres (yeah i know its a lot)
    TyreCompound current_compound;      // Current tyre of the car (C1-hardest - C5-softest)
    float tyre_temp_fl;                 // Front-Left
    float tyre_temp_fr;                 // Front-Right
    float tyre_temp_rl;                 // Rear-Left
    float tyre_temp_rr;                 // Rear-Right

    float tyre_wear_fl;                 // Front-Left
    float tyre_wear_fr;                 // Front-Right
    float tyre_wear_rl;                 // Rear-Left
    float tyre_wear_rr;                 // Rear-Right

    // battery and fuel
    float battery_mj;                   // Ammount of battery
    float fuel_kg;                      // Ammount of fuel

    // states for drs and modes
    bool drs_open;                      // drs boolean
    bool qualifying_mode;               // this is just cool to add for now but it states if its in qualifying mode or not, if it is then the car will not regenerate energy and will use more power to simulate a qualifying lap

    int laps_completed;                 // amount of laps completed by the sim for controlling better
};

// keeps the dynamics of the car each dt
struct F1CarDynamics {
    float total_mass;                   // mass_kg + fuel
    float pitch_angle;                  // track pitch

    float drag_force;                   // drag force
    float lateral_force;                // lateral force
    float gravity_longitudinal;         // gravity_longitudinal
    
    float max_grip;                     // max grip
    float long_grip;                    // longitudinal grip
    float max_traction_force;           // max traction force
    float engine_braking_force;         // engine braking

    float desired_braking_force;        // desired braking
    float desired_engine_force;         // desired power
    float applied_long_force;           // real f_long that pedals gave!

    float load_fl;                      // front left load
    float load_fr;                      // front right load
    float load_rl;                      // rear left load
    float load_rr;                      // rear right load
};

// track segment struct for track handling
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

// struct for results by cuda
struct SimResult {
    int setup_id;                       // Setup Id
    float lap_time;                     // Lap Time
    float top_speed_kmh;                // Top Speed
    float battery_used_mj;              // Ammount of Battery Used in MJ (starting + what is regenerated)
};

// struct to export telemetry data from best setup
struct TelemetryPoint {
    float speed_kmh;                    // Speed in segment in km/h
    float rpm;                          // RPM in segment
    int gear;                           // Gear in segment
    float throttle;                     // throttle (%)
    float brake;                        // brake (%)
    float time_s;                       // time s
    float tyre_temp_front;              // average temp front
    float tyre_temp_rear;               // average temp rear
};

// step physics for both visualizer and simulate lap cuda device code (CUDA_CALLABLE)
CUDA_CALLABLE void step_physics(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt);
// runs the simulate lap for N setups using the step_physics integration
void run_simulation_batch(const CarSetup* setups, SimResult* results, const TrackSegment* track, int num_segments, int numSetups);
// exports the best setup for csv
void export_simulated_telemetry(const CarSetup& best_setup, const TrackSegment* d_track, int num_segments, std::string year, std::string gp, std::string session);

#endif