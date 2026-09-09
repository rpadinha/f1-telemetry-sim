#ifndef HUD_RENDERER_H
#define HUD_RENDERER_H

#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include "physics.cuh"
#include "circuit_loader.h"

enum class HudMode {
    SIM_ONLY = 0,
    REAL_ONLY = 1,
    COMPARISON = 2
};

class HudManager {
private:
    sf::Font font;
    sf::Text txt_telemetry;
    sf::Text txt_timings;
    sf::Text txt_gear;

    sf::RectangleShape brake_bg, throttle_bg;
    sf::RectangleShape brake_fill, throttle_fill;
    sf::RectangleShape real_brake_fill, real_throttle_fill;

    sf::RectangleShape rpm_bg, rpm_fill, real_rpm_fill;

    float width, height;
    float pedal_w, pedal_h;
    float base_x, base_y;
    float hud_x, hud_y;


public:
    HudManager(float window_width, float window_height);
    bool load_font(const std::string& font_fath);

    void update(const F1Car& car, const std::vector<TrackSegment>& track, float s1, float s2, float s3, float last_lap_time, float sim_speed, HudMode mode);
    void draw(sf::RenderWindow& window, HudMode mode);
};

#endif