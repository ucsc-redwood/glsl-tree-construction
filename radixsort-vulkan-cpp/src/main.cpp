#include "app_params.hpp"
#include "naive_pipe.hpp"
#include <vulkan/vulkan.hpp>
#include <chrono>

#define BUFFER_ELEMENTS  131072

int main(const int argc, const char* argv[]){
    int n_blocks = 1;

    if (argc > 1){
        n_blocks = std::stoi(argv[1]);
    }
    AppParams app_params;
    app_params.n = BUFFER_ELEMENTS;
    app_params.min_coord = 0.0f;
    app_params.max_coord = 1.0f;
    app_params.seed = 114514;
    app_params.n_threads = 4;
    app_params.n_blocks = n_blocks;
    
   
    Pipe pipe = Pipe(app_params);
    pipe.allocate();
    
    pipe.init(n_blocks);
    
    pipe.morton(n_blocks);

    pipe.radix_sort(n_blocks);
    
    pipe.unique(n_blocks);

    //pipe.radix_tree(n_blocks);
    
    //pipe.edge_count(n_blocks);
    
    //pipe.prefix_sum(n_blocks);
    
    //pipe.octree(n_blocks);
    
    /*
    // step 0 initilization
    Init init_stage = Init();
    init_stage.run(app_params.n_blocks, data.data(), app_params.n, app_params.min_coord, app_params.getRange(), app_params.seed);
    */
    /*
    for (int i = 0; i < 1024; ++i){
        std::cout << data[i].x << " " << data[i].y << " " << data[i].z << " " << data[i].w << std::endl;
    }
    */
    
    /*
    // step 1 compute morton
    Morton morton_stage = Morton();
    morton_stage.run(app_params.n_blocks, data.data(), morton_keys.data(), app_params.n, app_params.min_coord, app_params.getRange());

    
    
    /*
    for(int i = 1; i <= BUFFER_ELEMENTS; ++i){
        morton_keys[i-1] = i;
    }
    */   
    
    /*
    // step 2 radix sort
    auto radixsort_stage = RadixSort();
    radixsort_stage.run(app_params.n_blocks, morton_keys.data(), app_params.n);


	for (int i = 0; i < 1024; i++){
		printf("sorted_key[%d]: %d\n", i, morton_keys[i]);
	}
    
    // step 3 remove duplicate
    auto unique_stage = Unique();
    unique_stage.run(app_params.n_blocks, morton_keys.data(), u_keys.data(), contribution.data() ,app_params.n);
    unique = contribution[app_params.n-1];
    n_brt_nodes = unique - 1;

    for (int i = 0; i < 1024; i++){
        printf("contribution[%d]: %d\n", i, contribution[i]);
    }
    for (int i = 0; i < 1024; i++){
        printf("u_keys[%d]: %d\n", i, u_keys[i]);
    }
    
    
    // step 4 build radix tree
    auto build_radix_tree_stage = RadixTree();
    build_radix_tree_stage.run(app_params.n_blocks, u_keys.data(), prefix_n.data(), has_leaf_left, has_leaf_right, left_child.data(), parents.data(), unique);

    for (int i = 61563; i < 61563+1024; i++){
        printf("prefix_n[%d]: %d\n", i, prefix_n[i]);
    }

    // step 5 edge count
    auto edge_count_stage = EdgeCount();
    edge_count_stage.run(app_params.n_blocks, prefix_n.data(), parents.data(),edge_count.data(), n_brt_nodes);
    
    for (int i = 0; i < 1024; i++){
        printf("edge_count[%d]: %d\n", i, edge_count[i]);
    }
    */
   /*
    
    for (int i = 0; i < app_params.n; ++i){
        edge_count[i] = 2;
    }
    
    // step 6 prefix sum
    auto prefix_sum_stage = PrefixSum();
    
    prefix_sum_stage.run(app_params.n_blocks, edge_count.data(), app_params.n);

    for (int i = 0; i < 1024; i++){
        printf("scanededge_count[%d]: %d\n", i, edge_count[i]);
    }
    */
    /*
    // step 7 build octree
    auto build_octree_stage = Octree();
    build_octree_stage.run(app_params.n_blocks,
    oct_nodes.data(),
    edge_count.data(),
    edge_count.data(),
    u_keys.data(),
    prefix_n.data(),
    has_leaf_left,
    has_leaf_right,
    parents.data(),
    left_child.data(),
    app_params.min_coord,
    app_params.getRange(),
    n_brt_nodes
    );
    */
    /*
    delete[] has_leaf_left;
    delete[] has_leaf_right;
    */
}