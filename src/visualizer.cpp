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

void reset_car_state(F1Car& car, const std::vector<TrackSegment>& track) {
    car.v = track[0].real_speed_kmh / 3.6f;
    car.battery_mj = Config::MAX_BATTERY_MJ;
    car.current_gear = track[0].real_gear;
    car.rpm = track[0].real_rpm;
    car.current_seg = 0;
    car.current_m = 0.0f;
    car.time_s = 0.0f;
    car.qualifying_mode = false;
    car.throttle_pedal = track[0].real_throttle_pedal;
    car.brake_pedal = track[0].real_brake_pedal;
    car.action = DriverAction::ACCELERATE;
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
    
    // the car shape init
    sf::CircleShape car_shape(6.0f);
    car_shape.setFillColor(sf::Color::Red);
    car_shape.setOrigin(6.0f, 6.0f);

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
    float sim_speed_multiplier = 1.0f;
    const int base_steps_per_frame = 8;
    int sector1_end = track.size() / 3;
    int sector2_end = (track.size() * 2) / 3;
    float time_s1 = 0.0f, time_s2 = 0.0f, time_s3 = 0.0f, last_lap_time = 0.0f;
    int prev_seg = 0;

    std::vector<sf::CircleShape> telemetry_trail;
    int frame_counter = 0;
    bool isPaused = false;
    HudMode current_mode = HudMode::SIM_ONLY;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
            
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::R) {
                    reset_car_state(car, track);
                    telemetry_trail.clear();
                    time_s1 = time_s2 = time_s3 = last_lap_time = 0.0f;
                }
                else if (event.key.code == sf::Keyboard::Up) {
                    if (sim_speed_multiplier == 0.1f) { sim_speed_multiplier += 0.4f; } // just cuz 0.1 + 0.5 = 0.6 and then on
                    else { sim_speed_multiplier += 0.5f; }
                }
                else if (event.key.code == sf::Keyboard::Down) {
                    sim_speed_multiplier = std::max(0.1f, sim_speed_multiplier - 0.5f);
                }
                else if (event.key.code == sf::Keyboard::Space) {
                    isPaused = !isPaused;
                }
                else if (event.key.code == sf::Keyboard::Right) {
                    current_mode = static_cast<HudMode>((static_cast<int>(current_mode) + 1) % 3);
                }
                else if (event.key.code == sf::Keyboard::Left) {
                    current_mode = static_cast<HudMode>((static_cast<int>(current_mode) + 2) % 3);
                }
            }
        }

        if (!isPaused) {
            int current_steps = std::max(1, static_cast<int>(base_steps_per_frame * sim_speed_multiplier));
            for (int i = 0; i < current_steps; ++i) {
                int old_seg = car.current_seg;

                step_physics(&car, &setup, track.data(), track.size(), physics_dt);

                if (old_seg < sector1_end && car.current_seg >= sector1_end) {
                    time_s1 = car.time_s;
                } else if (old_seg < sector2_end && car.current_seg >= sector2_end) {
                    time_s2 = car.time_s - time_s1;
                }

                if (car.current_seg >= track.size() - 1) {
                    time_s3 = car.time_s - (time_s1 + time_s2);
                    last_lap_time = car.time_s;
                    
                    reset_car_state(car, track);
                    telemetry_trail.clear();
                }
            }
        }

        int next_idx = (car.current_seg + 1) % track.size();
        
        float ax = track[car.current_seg].x;
        float ay = track[car.current_seg].y;
        float bx = track[next_idx].x;
        float by = track[next_idx].y;
        float seg_length = track[car.current_seg].length_m;
        float dir_x = (bx - ax) / seg_length;
        float dir_y = (by - ay) / seg_length;
        
        float real_car_x = ax + (dir_x * car.current_m);
        float real_car_y = ay + (dir_y * car.current_m);
        
        // Usamos as variáveis matemáticas que vieram no pacote t_data
        float screen_car_x = t_data.offset_x + (real_car_x - t_data.min_x) * t_data.scale;
        float screen_car_y = t_data.offset_y + (t_data.max_y - real_car_y) * t_data.scale;
        
        sf::Color current_color = (car.action == DriverAction::ACCELERATE) ? sf::Color::Green :
                                (car.action == DriverAction::BRAKE) ? sf::Color::Red : sf::Color::Yellow;
        car_shape.setFillColor(current_color);

        frame_counter++;
        if (frame_counter % 3 == 0) {
            sf::CircleShape dot(2.5f);
            dot.setFillColor(current_color);
            dot.setOrigin(2.5f, 2.5f);
            dot.setPosition(screen_car_x, screen_car_y);
            telemetry_trail.push_back(dot);
        }

        car_shape.setPosition(screen_car_x, screen_car_y);

        window.clear(sf::Color(20, 20, 20));

        window.draw(t_data.track_lines);

        for (const auto& dot : telemetry_trail) {
            window.draw(dot);
        }
        window.draw(car_shape);

        hud.update(car, track, time_s1, time_s2, time_s3, last_lap_time, sim_speed_multiplier, current_mode);
        hud.draw(window, current_mode);
        
        window.display();
    }
}