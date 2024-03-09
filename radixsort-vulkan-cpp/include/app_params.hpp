#pragma once

struct AppParams {

  [[nodiscard]] float getRange() { return max_coord - min_coord; }

  int n;
  float min_coord;
  float max_coord;
  int seed;
  int n_threads;
  int n_blocks;
};