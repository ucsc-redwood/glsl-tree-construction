#pragma once

#include "application.hpp"
#include <glm/vec4.hpp>

class Pipe : public ApplicationBase {
 public:
  Pipe() : ApplicationBase() {}
  void allocate(const int n);
  ~Pipe();
  protected:
  static constexpr auto educated_guess_nodes = 0.6f;
  int n_;

  int n_pts;
  int n_unique_keys;
  int n_brt_nodes;  // unique_keys - 1
  int n_oct_nodes;  // computed late... we use 0.6 * n as a guess

  // Essential data memory
  std::vector<glm::vec4> u_points;
  std::vector<unsigned int> u_morton_keys;
  std::vector<unsigned int> u_unique_morton_keys;
  std::vector<int> u_edge_count;
  std::vector<int> u_edge_offset;

  // Essential
  // should be of size 'n_unique_keys - 1', but we can allocate as 'n' for now
  struct {
    std::vector<uint8_t> u_prefix_n;
    bool* u_has_leaf_left;  // you can't use vector of bools
    bool* u_has_leaf_right;
    std::vector<int> u_left_child;
    std::vector<int> u_parent;
  } brt;

  // Essential
  // should be of size 'n_oct_nodes', we use an educated guess for now
  struct {
    int (*u_children)[8];
    glm::vec4* u_corner;
    float* u_cell_size;
    int* u_child_node_mask;
    int* u_child_leaf_mask;
  } oct;

  // Temp
  struct {
    std::vector<unsigned int> u_sort_alt;              // n
    std::vector<unsigned int> u_global_histogram;      // 256 * 4
    std::vector<unsigned int> u_index;                 // 4
    std::vector<unsigned int> u_first_pass_histogram;  // 256 * xxx
    std::vector<unsigned int> u_second_pass_histogram;
    std::vector<unsigned int> u_third_pass_histogram;
    std::vector<unsigned int> u_fourth_pass_histogram;
  } sort_tmp;

  struct {
    std::vector<int> u_flag_heads;  // n
  } unique_tmp;

  struct {
    // use Agent's tile size to allocate
    std::vector<int> u_auxiliary;  // n_tiles
  } prefix_sum_tmp;
  
};


void Pipe::allocate(const int n) {
    n_ = n;
    // --- Essentials ---
    u_points.resize(n);
    u_morton_keys.resize(n);
    u_unique_morton_keys.resize(n);

    brt.u_prefix_n.resize(n);  // should be n_unique, but n will do for now
    MallocManaged(&brt.u_has_leaf_left, n);
    MallocManaged(&brt.u_has_leaf_right, n);
    brt.u_left_child.resize(n);
    brt.u_parent.resize(n);

    u_edge_count.resize(n);
    u_edge_offset.resize(n);

    const auto num_oct_to_allocate = n * educated_guess_nodes;
    MallocManaged(&oct.u_children, num_oct_to_allocate);
    MallocManaged(&oct.u_corner, num_oct_to_allocate);
    MallocManaged(&oct.u_cell_size, num_oct_to_allocate);
    MallocManaged(&oct.u_child_node_mask, num_oct_to_allocate);
    MallocManaged(&oct.u_child_leaf_mask, num_oct_to_allocate);

    // -------------------------

    // Temporary storages for Sort
    constexpr auto radix = 256;
    constexpr auto passes = 4;
    const auto binning_thread_blocks = n + 7680 /;
    sort_tmp.u_sort_alt.resize(n);
    sort_tmp.u_global_histogram.resize(radix * passes);
    sort_tmp.u_index.resize(passes);
    sort_tmp.u_first_pass_histogram.resize(radix * binning_thread_blocks);
    sort_tmp.u_second_pass_histogram.resize(radix * binning_thread_blocks);
    sort_tmp.u_third_pass_histogram.resize(radix * binning_thread_blocks);
    sort_tmp.u_fourth_pass_histogram.resize(radix * binning_thread_blocks);

    // Temporary storages for Unique
    unique_tmp.u_flag_heads.resize(n);

    // Temporary storages for PrefixSum
    constexpr auto prefix_sum_tile_size = gpu::PrefixSumAgent<int>::tile_size;
    const auto prefix_sum_n_tiles =
        cub::DivideAndRoundUp(n, prefix_sum_tile_size);
    prefix_sum_tmp.u_auxiliary.resize(prefix_sum_n_tiles);
};

~Pipe() {
  // --- Essentials ---
  u_points.clear();
  u_morton_keys.clear();
  u_unique_morton_keys.clear();

  brt.u_prefix_n.clear();
  FreeManaged(brt.u_has_leaf_left);
  FreeManaged(brt.u_has_leaf_right);
  brt.u_left_child.clear();
  brt.u_parent.clear();

  u_edge_count.clear();
  u_edge_offset.clear();

  FreeManaged(oct.u_children);
  FreeManaged(oct.u_corner);
  FreeManaged(oct.u_cell_size);
  FreeManaged(oct.u_child_node_mask);
  FreeManaged(oct.u_child_leaf_mask);

  // -------------------------

  // Temporary storages for Sort
  sort_tmp.u_sort_alt.clear();
  sort_tmp.u_global_histogram.clear();
  sort_tmp.u_index.clear();
  sort_tmp.u_first_pass_histogram.clear();
  sort_tmp.u_second_pass_histogram.clear();
  sort_tmp.u_third_pass_histogram.clear();
  sort_tmp.u_fourth_pass_histogram.clear();

  // Temporary storages for Unique
  unique_tmp.u_flag_heads.clear();

  // Temporary storages for PrefixSum
  prefix_sum_tmp.u_auxiliary.clear();
}