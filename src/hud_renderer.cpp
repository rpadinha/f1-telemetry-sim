#include "hud_renderer.hpp"
#include <algorithm>

HudManager::HudManager(float window_width, float window_height) 
    : width(window_width), height(window_height), 
      pedal_w(30.f), pedal_h(100.f), 
      base_x(20.f), base_y(height - 20.f),
      hud_x(width / 2.0f - 200.f), hud_y(height - 50.f)
{
    txt_telemetry.setCharacterSize(18);
    txt_telemetry.setFillColor(sf::Color::White);
    txt_telemetry.setPosition(20, 20);

    txt_timings.setCharacterSize(18);
    txt_timings.setFillColor(sf::Color::White);
    txt_timings.setPosition(1100, 20);

    txt_gear.setCharacterSize(24);
    txt_gear.setFillColor(sf::Color::White);
    txt_gear.setPosition(hud_x + 100.f, hud_y - 35.f);

    // Backgrounds dos Pedais
    brake_bg.setSize(sf::Vector2f(pedal_w, -pedal_h));
    brake_bg.setPosition(base_x, base_y);
    brake_bg.setFillColor(sf::Color(50, 50, 50, 200));

    throttle_bg.setSize(sf::Vector2f(pedal_w, -pedal_h));
    throttle_bg.setPosition(base_x + 40.f, base_y);
    throttle_bg.setFillColor(sf::Color(50, 50, 50, 200));

    // Background RPM
    rpm_bg.setSize(sf::Vector2f(400.f, 20.f));
    rpm_bg.setPosition(hud_x, hud_y);
    rpm_bg.setFillColor(sf::Color(50, 50, 50, 200));

    brake_fill.setPosition(base_x, base_y);
    throttle_fill.setPosition(base_x + 40.f, base_y);
    real_brake_fill.setPosition(base_x, base_y);
    real_throttle_fill.setPosition(base_x + 40.f, base_y);
    rpm_fill.setPosition(hud_x, hud_y);
    real_rpm_fill.setPosition(hud_x, hud_y);
}

bool HudManager::load_font(const std::string& font_path) {
    if (!font.loadFromFile(font_path)) return false;
    txt_telemetry.setFont(font);
    txt_timings.setFont(font);
    txt_gear.setFont(font);
    return true;
}

void HudManager::update(const F1Car& car, const std::vector<TrackSegment>& track, 
                        float s1, float s2, float s3, float last_lap_time, 
                        float sim_speed, HudMode mode) {
    
    float real_throttle = track[car.current_seg].real_throttle_pedal;
    float real_brake = track[car.current_seg].real_brake_pedal;
    
    std::string mode_title;
    switch(mode) {
        case HudMode::SIM_ONLY: mode_title = "MODE: SIM DRIVER"; break;
        case HudMode::REAL_ONLY: mode_title = "MODE: REAL DRIVER"; break;
        case HudMode::COMPARISON: mode_title = "MODE: COMPARISON (SIM-Solid) (REAL-Outline)"; break;
    }

    float sim_speed_kmh = car.v * 3.6f;
    float delta_speed = sim_speed_kmh - track[car.current_seg].real_speed_kmh;
    float delta_rpm = car.rpm - track[car.current_seg].real_rpm;

    char buf_tel[512];
    snprintf(buf_tel, sizeof(buf_tel),
        "%s\n\n"
        "MGU-K: %.2f MJ\n"
        "Sim Speed: %.1fx\n\n"
        "--- TRACK ---\n"
        "Seg: %d / %lu\n"
        "Radius: %s\n",
        mode_title.c_str(), car.battery_mj, sim_speed,
        car.current_seg, track.size(),
        (track[car.current_seg].radius_m >= 10000.f) ? "STRAIGHT" : std::to_string((int)track[car.current_seg].radius_m).c_str());
    txt_telemetry.setString(buf_tel);

    char buf_timing[256];
    snprintf(buf_timing, sizeof(buf_timing), "S1: %.3f\nS2: %.3f\nS3: %.3f\nLap Time: %.3f\nLast Lap: %.3f", s1, s2, s3, car.time_s, last_lap_time);
    txt_timings.setString(buf_timing);

    char gear_buf[64];
    if (mode == HudMode::SIM_ONLY) {
        snprintf(gear_buf, sizeof(gear_buf), "GEAR: %d   %3.0f KM/H", car.current_gear, sim_speed_kmh);
    } else if (mode == HudMode::REAL_ONLY) {
        snprintf(gear_buf, sizeof(gear_buf), "GEAR: %d   %3.0f KM/H", track[car.current_seg].real_gear, track[car.current_seg].real_speed_kmh);
    } else {
        snprintf(gear_buf, sizeof(gear_buf), "GEAR: %d/%d   %3.0f/%3.0f KM/H", car.current_gear, track[car.current_seg].real_gear, sim_speed_kmh, track[car.current_seg].real_speed_kmh);
    }
    txt_gear.setString(gear_buf);

    // Atualizar tamanhos e cores (IA)
    brake_fill.setSize(sf::Vector2f(pedal_w, -(pedal_h * car.brake_pedal)));
    brake_fill.setFillColor(sf::Color(220, 50, 50));
    throttle_fill.setSize(sf::Vector2f(pedal_w, -(pedal_h * car.throttle_pedal)));
    throttle_fill.setFillColor(sf::Color(50, 220, 50));

    real_brake_fill.setSize(sf::Vector2f(pedal_w, -(pedal_h * real_brake)));
    real_throttle_fill.setSize(sf::Vector2f(pedal_w, -(pedal_h * real_throttle)));

    if (mode == HudMode::COMPARISON) {
        real_brake_fill.setFillColor(sf::Color::Transparent);
        real_brake_fill.setOutlineThickness(-2.f);
        real_brake_fill.setOutlineColor(sf::Color::White);
        
        real_throttle_fill.setFillColor(sf::Color::Transparent);
        real_throttle_fill.setOutlineThickness(-2.f);
        real_throttle_fill.setOutlineColor(sf::Color::White);
    } else {
        real_brake_fill.setFillColor(sf::Color(255, 140, 0)); 
        real_throttle_fill.setFillColor(sf::Color(0, 200, 255));
        real_brake_fill.setOutlineThickness(0.f);
        real_throttle_fill.setOutlineThickness(0.f);
    }

    // Atualizar RPM
    float rpm_pct = std::min(car.rpm / 12500.0f, 1.0f);
    rpm_fill.setSize(sf::Vector2f(400.f * rpm_pct, 20.f));
    rpm_fill.setFillColor(car.rpm > 11800.0f ? sf::Color::Red : sf::Color::Green);

    float real_rpm_pct = std::min(track[car.current_seg].real_rpm / 12500.0f, 1.0f);
    real_rpm_fill.setSize(sf::Vector2f(400.f * real_rpm_pct, 20.f));
    if (mode == HudMode::COMPARISON) {
        real_rpm_fill.setFillColor(sf::Color::Transparent);
        real_rpm_fill.setOutlineThickness(-2.f);
        real_rpm_fill.setOutlineColor(sf::Color::White);
    } else {
        real_rpm_fill.setFillColor(sf::Color(0, 200, 255));
        real_rpm_fill.setOutlineThickness(0.f);
    }
}

void HudManager::draw(sf::RenderWindow& window, HudMode mode) {
    window.draw(txt_telemetry);
    window.draw(txt_timings);
    window.draw(brake_bg);
    window.draw(throttle_bg);
    window.draw(rpm_bg);
    window.draw(txt_gear);

    if (mode == HudMode::SIM_ONLY || mode == HudMode::COMPARISON) {
        window.draw(brake_fill);
        window.draw(throttle_fill);
        window.draw(rpm_fill);
    }
    if (mode == HudMode::REAL_ONLY || mode == HudMode::COMPARISON) {
        window.draw(real_brake_fill);
        window.draw(real_throttle_fill);
        window.draw(real_rpm_fill);
    }
}