#include "visualizer.h"
#include "config.cuh"
#include "hud_renderer.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <algorithm>
#include <math.h>

#define WIDTH 1280
#define HEIGHT 720

struct TrackRenderData {
    float min_x, max_y;
    float offset_x, offset_y;
    float scale;
    sf::VertexArray track_lines;
};

struct SimState {
    float sim_speed_multiplier = 1.0f;
    float time_s1 = 0.0f, time_s2 = 0.0f, time_s3 = 0.0f, last_lap_time = 0.0f;
    int frame_counter = 0;
    bool isPaused = false;
    HudMode current_mode = HudMode::SIM_ONLY;
    std::vector<sf::CircleShape> telemetry_trail;
};

std::vector<float> build_real_time(const std::vector<TrackSegment>& track) {
    std::vector<float> times(track.size(), 0.0f);
    float current_time = 0.0f;

    for (size_t i = 0; i < track.size() ; ++i) {
        times[i] = current_time;
        float v_ms = track[i].real_speed_kmh / 3.6f;
        if (v_ms < 1.0f) v_ms = 1.0f;

        current_time += track[i].length_m / v_ms;
    }
    return times;
}

void get_real_driver_position(float sim_time, const std::vector<float>& times, const std::vector<TrackSegment>& track, int& out_seg, float& out_m) {
    auto it = std::upper_bound(times.begin(), times.end(), sim_time);

    if (it == times.begin()) {
        out_seg = 0; out_m = 0.0f; return;
    }
    if (it == times.end()) {
        out_seg = track.size() - 1;
        out_m = track.back().length_m; return;
    }

    int idx = std::distance(times.begin(), it) - 1;
    out_seg = idx;

    float time_spent_in_geg = sim_time - times[idx];
    float v_ms = track[idx].real_speed_kmh / 3.6f;

    out_m = time_spent_in_geg * v_ms;
    if (out_m > track[idx].length_m) out_m = track[idx].length_m;
}

TrackRenderData build_track_shapes(const std::vector<TrackSegment>& track) {
    TrackRenderData data;
    data.track_lines.setPrimitiveType(sf::LineStrip);
    data.track_lines.resize(track.size());

    float min_x = track[0].x, max_x = track[0].x;
    float min_y = track[0].y, max_y = track[0].y;

    for (const auto& pt : track) {
        if (pt.x < min_x) min_x = pt.x;
        if (pt.x > max_x) max_x = pt.x;
        if (pt.y < min_y) min_y = pt.y;
        if (pt.y > max_y) max_y = pt.y;
    }

    data.min_x = min_x;
    data.max_y = max_y;

    float margin = 50.0f;
    float scale_x = (WIDTH - 2.0f * margin) / (max_x - min_x);
    float scale_y = (HEIGHT - 2.0f * margin) / (max_y - min_y);
    data.scale = std::min(scale_x, scale_y);
    
    data.offset_x = (WIDTH - ((max_x - min_x) * data.scale)) / 2.0f;
    data.offset_y = (HEIGHT - ((max_y - min_y) * data.scale)) / 2.0f;

    for (size_t i = 0; i < track.size(); ++i) {
        float screen_x = data.offset_x + (track[i].x - min_x) * data.scale;
        float screen_y = data.offset_y + (max_y - track[i].y) * data.scale;
        data.track_lines[i].position = sf::Vector2f(screen_x, screen_y);
        data.track_lines[i].color = sf::Color(150, 150, 150);
    }
    
    return data;
}

sf::Vector2f get_screen_coordinates(int current_seg, float current_m, const std::vector<TrackSegment>& track, const TrackRenderData& t_data) {
    int next_idx = (current_seg + 1) % track.size();
    
    float ax = track[current_seg].x;
    float ay = track[current_seg].y;
    float bx = track[next_idx].x;
    float by = track[next_idx].y;
    
    float seg_length = track[current_seg].length_m;
    if (seg_length <= 0.0f) seg_length = 1.0f;
    
    float dir_x = (bx - ax) / seg_length;
    float dir_y = (by - ay) / seg_length;
    
    float real_x = ax + (dir_x * current_m);
    float real_y = ay + (dir_y * current_m);
    
    float screen_x = t_data.offset_x + (real_x - t_data.min_x) * t_data.scale;
    float screen_y = t_data.offset_y + (t_data.max_y - real_y) * t_data.scale;
    
    return sf::Vector2f(screen_x, screen_y);
}

void reset_car_state(F1Car& car, const std::vector<TrackSegment>& track) {
    car.v = track[0].real_speed_kmh / 3.6f;
    car.battery_mj = Config::MAX_BATTERY_MJ;
    car.current_gear = track[0].real_gear;
    car.rpm = track[0].real_rpm;
    car.current_seg = 0;
    car.current_m = 0.0f;
    car.time_s = 0.0f;
    car.qualifying_mode = true;
    car.throttle_pedal = track[0].real_throttle_pedal;
    car.brake_pedal = track[0].real_brake_pedal;
    car.action = DriverAction::ACCELERATE;
}

void handle_user_inputs(sf::RenderWindow& window, sf::Event& event, F1Car& car, const std::vector<TrackSegment>& track, SimState& state) {
    if (event.type == sf::Event::Closed) {
        window.close();
    }
    else if (event.type == sf::Event::KeyPressed) {
        if (event.key.code == sf::Keyboard::R) {
            reset_car_state(car, track);
            state.telemetry_trail.clear();
            state.time_s1 = state.time_s2 = state.time_s3 = state.last_lap_time = 0.0f;
        }
        else if (event.key.code == sf::Keyboard::Up) {
            if (state.sim_speed_multiplier <= 0.1f) state.sim_speed_multiplier += 0.4f; 
            else state.sim_speed_multiplier += 0.5f;
        }
        else if (event.key.code == sf::Keyboard::Down) {
            state.sim_speed_multiplier = std::max(0.1f, state.sim_speed_multiplier - 0.5f);
        }
        else if (event.key.code == sf::Keyboard::Space) {
            state.isPaused = !state.isPaused;
        }
        else if (event.key.code == sf::Keyboard::Right) {
            state.current_mode = static_cast<HudMode>((static_cast<int>(state.current_mode) + 1) % 3);
        }
        else if (event.key.code == sf::Keyboard::Left) {
            state.current_mode = static_cast<HudMode>((static_cast<int>(state.current_mode) + 2) % 3);
        }
    }
}

void update_simulation_step(F1Car& car, const CarSetup& setup, const std::vector<TrackSegment>& track, SimState& state, float physics_dt, float base_steps_per_frame, bool isPaused) {
    int sector1_end = track.size() / 3;
    int sector2_end = (track.size() * 2) / 3;
    if (!state.isPaused) {
        int current_steps = std::max(1, static_cast<int>(base_steps_per_frame * state.sim_speed_multiplier));

        for (int i = 0; i < current_steps; ++i) {
            int old_seg = car.current_seg;
            step_physics(&car, &setup, track.data(), track.size(), physics_dt);

            if (old_seg < sector1_end && car.current_seg >= sector1_end) {
                    state.time_s1 = car.time_s;
            } else if (old_seg < sector2_end && car.current_seg >= sector2_end) {
                    state.time_s2 = car.time_s - state.time_s1;
            }

            if (car.current_seg >= track.size() - 1) {
                state.time_s3 = car.time_s - (state.time_s1 + state.time_s2);
                state.last_lap_time = car.time_s;
                reset_car_state(car, track);
                state.telemetry_trail.clear();
                }
            }
        }
}

void run_sfml_visualizer(const std::vector<TrackSegment>& track, const CarSetup& setup) {

    std::cout << "[SFML] Getting track limits..." << std::endl;
    // Track getter
    TrackRenderData t_data = build_track_shapes(track);

    // Windows Creation
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "F1 Telemetry Visualizer");
    window.setFramerateLimit(60);
    
    // the car shapes init
    sf::CircleShape car_shape(6.0f);
    car_shape.setFillColor(sf::Color::Red);
    car_shape.setOrigin(6.0f, 6.0f);

    sf::CircleShape real_car_shape(7.0f); 
    real_car_shape.setFillColor(sf::Color::Transparent);
    real_car_shape.setOutlineThickness(2.0f);
    real_car_shape.setOutlineColor(sf::Color::Cyan);
    real_car_shape.setOrigin(7.0f, 7.0f);
    
    // ADICIONAR ISTO: Calcular a linha temporal
    std::cout << "[SFML] Building real driver timeline..." << std::endl;
    std::vector<float> real_timeline = build_real_time(track);

    // Text and telemetry initialization
    HudManager hud(WIDTH, HEIGHT);
    if (!hud.load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
        std::cerr << "[SFML] Font not found!" << std::endl;
        return;
    }

    // car init
    F1Car car;
    reset_car_state(car, track);
    
    // physics and timer init
    // (2ms per step)
    float physics_dt = 0.002f;
    const int base_steps_per_frame = 8;
    int sector1_end = track.size() / 3;
    int sector2_end = (track.size() * 2) / 3;
    SimState state;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            handle_user_inputs(window, event, car, track, state);
        }

        update_simulation_step(car, setup, track, state, physics_dt, base_steps_per_frame, state.isPaused);

        sf::Vector2f car_pos = get_screen_coordinates(car.current_seg, car.current_m, track, t_data);
        
        sf::Color current_color = (car.action == DriverAction::ACCELERATE) ? sf::Color::Green :
                                (car.action == DriverAction::BRAKE) ? sf::Color::Red : sf::Color::Yellow;
        car_shape.setFillColor(current_color);

        state.frame_counter++;
        if (state.frame_counter % 3 == 0) {
            sf::CircleShape dot(2.5f);
            dot.setFillColor(current_color);
            dot.setOrigin(2.5f, 2.5f);
            dot.setPosition(car_pos);
            state.telemetry_trail.push_back(dot);
        }

        car_shape.setPosition(car_pos);

        window.clear(sf::Color(20, 20, 20));

        window.draw(t_data.track_lines);

        for (const auto& dot : state.telemetry_trail) { window.draw(dot); }

        if (state.current_mode == HudMode::SIM_ONLY || state.current_mode == HudMode::COMPARISON) {
            sf::Vector2f car_pos = get_screen_coordinates(car.current_seg, car.current_m, track, t_data);
            
            sf::Color current_color = (car.action == DriverAction::ACCELERATE) ? sf::Color::Green :
                                    (car.action == DriverAction::BRAKE) ? sf::Color::Red : sf::Color::Yellow;
            car_shape.setFillColor(current_color);

            state.frame_counter++;
            if (state.frame_counter % 3 == 0 && !state.isPaused) {
                sf::CircleShape dot(2.5f);
                dot.setFillColor(current_color);
                dot.setOrigin(2.5f, 2.5f);
                dot.setPosition(car_pos);
                state.telemetry_trail.push_back(dot);
            }

            car_shape.setPosition(car_pos);
            window.draw(car_shape);
        }

        if (state.current_mode == HudMode::REAL_ONLY || state.current_mode == HudMode::COMPARISON) {
            int real_seg; 
            float real_m;
            
            get_real_driver_position(car.time_s, real_timeline, track, real_seg, real_m);
            
            sf::Vector2f real_pos = get_screen_coordinates(real_seg, real_m, track, t_data);
            
            real_car_shape.setPosition(real_pos);
            window.draw(real_car_shape);
        }

        hud.update(car, track, state.time_s1, state.time_s2, state.time_s3, state.last_lap_time, state.sim_speed_multiplier, state.current_mode);
        hud.draw(window, state.current_mode);
        
        window.display();
    }
}