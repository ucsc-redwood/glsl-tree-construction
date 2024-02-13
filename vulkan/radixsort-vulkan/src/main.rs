// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to use the compute capabilities of Vulkan.
//
// While graphics cards have traditionally been used for graphical operations, over time they have
// been more or more used for general-purpose operations as well. This is called "General-Purpose
// GPU", or *GPGPU*. This is what this example demonstrates.
use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::Vector4;
use rand::Rng;
use std::sync::Arc;
use std::vec;
use std::{fs::File, io::Write};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, pool::CommandPoolResetFlags,
        AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{self},
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features,
        QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    shader::ShaderStages,
    sync::{self, GpuFuture},
    DeviceSize, VulkanLibrary,
};
mod build_radix_tree;
mod edge_count;
mod init;
mod morton;
mod octree;
mod partial_sum;
mod radixsort;
mod unique;

struct RadixTreeData {
    n_nodes: u32,
    prefix_n: Vec<u8>,
    has_leaf_left: Vec<u8>,
    has_leaf_right: Vec<u8>,
    left_child: Vec<i32>,
    parent: Vec<i32>,
}

const N: u32 = 10000000;
const MORTON_BIT: u32 = 30;

fn main() {
    /* 
    let min: f32 = 0.0;
    let max: f32 = 1024.0;
    let range = max - min;
    let mut u_sort: Vec<u32> = vec![0; N as usize];
    let mut out_data: Vec<u32> = vec![0; N as usize];
    let mut u_input: Vec<[f32; 4]> = vec![[0.0; 4]; N as usize];
    init::init_random(&mut u_input, N, min, range);
    /* 
    for n in 0..N {
        println!("u_input[{}]: {:?}", n, u_input[n as usize]);
    }
    */
    // step 1
    morton::compute_morton(u_input, &mut u_sort, N, min, range, 256);
    /* 
    for n in 0..N {
        println!("u_sort[{}]: {}", n, u_sort[n as usize]);
    }
    */
    // step 2
    /* 
    radixsort::radix_sort(u_sort, N, &mut out_data, 1);
    */
     
    u_sort.sort();
    out_data = u_sort.clone();
    /*
    for n in 0..N {
        println!("out_data[{}]: {}", n, out_data[n as usize]);
    }
    */
    // step 3
    let n_unique = unique::unique(&mut out_data, N);

    println!("n_unique: {}", n_unique);

    let mut tree = RadixTreeData {
        n_nodes: n_unique - 1,
        prefix_n: vec![0; (n_unique - 1).try_into().unwrap()],
        has_leaf_left: vec![0; (n_unique - 1).try_into().unwrap()],
        has_leaf_right: vec![0; (n_unique - 1).try_into().unwrap()],
        left_child: vec![0; (n_unique - 1).try_into().unwrap()],
        parent: vec![0; (n_unique - 1).try_into().unwrap()],
    };

    build_radix_tree::build_radix_tree(
        tree.n_nodes as i32,
        &out_data,
        &mut tree.prefix_n,
        &mut tree.has_leaf_left,
        &mut tree.has_leaf_right,
        &mut tree.left_child,
        &mut tree.parent,
        256,
    );

    let mut u_edge_count = vec![0; tree.n_nodes as usize];

    // step 4
    edge_count::edge_count(&tree.prefix_n, &tree.parent, &mut u_edge_count, n_unique, 256);
    /* 
    for n in 0..n_unique - 1 {
        println!("u_edge_count[{}]: {}", n, u_edge_count[n as usize]);
    }
    */
    // step 5
    let mut u_count_prefix_sum: Vec<u32> = vec![0; n_unique as usize];
    partial_sum::partial_sum(&u_edge_count, 0, n_unique, &mut u_count_prefix_sum);
    u_count_prefix_sum[0] = 0;
    /*
    for n in 0..n_unique - 1 {
        println!(
            "u_count_prefix_sum[{}]: {}",
            n, u_count_prefix_sum[n as usize]
        );
    }
    */

    // step 6
    let mut u_oct_nodes = vec![octree::OctNode::new(); n_unique as usize];

    let root_level: u32 = (tree.prefix_n[0] / 3).into();
    let root_prefix: u32 = out_data[0] >> (MORTON_BIT - root_level * 3);

    let mut corner: [f32; 4] = [0.0; 4];
    morton::morton32_to_xyz(
        &mut corner,
        root_prefix << (MORTON_BIT - (3 * root_level)),
        min,
        range,
    );
    u_oct_nodes[0].set_corner(corner);
    u_oct_nodes[0].set_cell_size(range);

    octree::build_octree(
        &mut u_oct_nodes,
        u_count_prefix_sum,
        u_edge_count,
        out_data,
        tree.prefix_n,
        tree.parent,
        min,
        range,
        N as i32,
        tree.has_leaf_left,
        tree.has_leaf_right,
        tree.left_child,
        256,
    )
    */
    test_radix_sort();
}

fn test_radix_sort() {
    // initialize the data
    let mut rng = rand::thread_rng();
    let mut random_numbers: Vec<u32> = (0..15360 / 2).collect::<Vec<u32>>(); /*map(|_| rng.gen()).collect()*/
    random_numbers.reverse();
    //let mut random_numbers: Vec<[f32; 4]> = [[0.0; 4]; 15360].to_vec();
    //random_numbers.reverse();
    //init::init_random(&mut random_numbers);
    for n in 0..15360 / 2 {
        println!("coords: {:?}", random_numbers[n as usize]);
    }
    /*
    let morton_keys = morton::compute_morton(random_numbers);
    for n in 0..15360/2 {
        println!("morton_keys[{}]: {}", n, morton_keys[n as usize]);
    }
    */

    let pass_hist = vec![0 as u32; 1024];
    let b_globalHist = vec![0 as u32; 256 * 4];

    // As with other examples, the first step is to create an instance.
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            // flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    // Choose which physical device to use.
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };
    let device_features = Features {
        //subgroup_size_control: true,
        shader_int64: true,
        ..Features::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
                && p.supported_features().contains(&device_features)
        })
        .filter_map(|p| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one
            // queue that supports compute operations.
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    // Now initializing the device.
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_features: Features {
                shader_int64: true,
                ..Features::empty()
            },
            ..Default::default()
        },
    )
    .unwrap();

    if !device
        .physical_device()
        .properties()
        .required_subgroup_size_stages
        .unwrap_or_default()
        .intersects(ShaderStages::COMPUTE)
    {
        println!("Subgroup size control is not supported for compute shaders");
        //return;
    }

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retrieve the first and only element of the
    // iterator and throw it away.
    let queue = queues.next().unwrap();

    // Now let's get to the actual example.
    //
    // What we are going to do is very basic: we are going to fill a buffer with 64k integers and
    // ask the GPU to multiply each of them by 12.
    //
    // GPUs are very good at parallel computations (SIMD-like operations), and thus will do this
    // much more quickly than a CPU would do. While a CPU would typically multiply them one by one
    // or four by four, a GPU will do it by groups of 32 or 64.
    //
    // Note however that in a real-life situation for such a simple operation the cost of accessing
    // memory usually outweighs the benefits of a faster calculation. Since both the CPU and the
    // GPU will need to access data, there is no other choice but to transfer the data through the
    // slow PCI express bus.

    // We need to create the compute pipeline that describes our operation.
    //
    // If you are familiar with graphics pipeline, the principle is the same except that compute
    // pipelines are much simpler to create.

    mod cs {

        vulkano_shaders::shader! {
            ty: "compute",
            path: "shader/new_radix_sort.comp",
            spirv_version: "1.3",
        }
    }

    let cs = cs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let stage = PipelineShaderStageCreateInfo {
        //required_subgroup_size: Some(32),
        ..PipelineShaderStageCreateInfo::new(cs)
    };
    let pipeline = {
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap()
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    // We start by creating the buffer that will store the data.
    let b_sort_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        random_numbers, //morton_keys,
    )
    .unwrap();

    let b_alt_buffer = Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        15360 / 2 as DeviceSize,
    )
    .unwrap();

    let b_globalhist_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        b_globalHist,
    )
    .unwrap();

    let b_index_buffer = Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        4 as DeviceSize,
    )
    .unwrap();

    let b_passhist_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        pass_hist,
    )
    .unwrap();

    let out_reduction_buffer = Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        256 as DeviceSize,
    )
    .unwrap();

    // In order to let the shader access the buffer, we need to build a *descriptor set* that
    // contains the buffer.
    //
    // The resources that we bind to the descriptor set must match the resources expected by the
    // pipeline which we pass as the first parameter.
    //
    // If you want to run the pipeline on multiple different buffers, you need to create multiple
    // descriptor sets that each contain the buffer you want to run the shader on.
    let layout = pipeline.layout().set_layouts().first().unwrap();

    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, b_sort_buffer.clone()),
            WriteDescriptorSet::buffer(1, b_alt_buffer.clone()),
            WriteDescriptorSet::buffer(2, b_globalhist_buffer.clone()),
            WriteDescriptorSet::buffer(3, b_index_buffer.clone()),
            WriteDescriptorSet::buffer(4, b_passhist_buffer.clone()),
            //WriteDescriptorSet::buffer(5, out_reduction_buffer.clone()),
        ],
        [],
    )
    .unwrap();

    command_buffer_allocator
        .try_reset_pool(
            queue.queue_family_index(),
            CommandPoolResetFlags::RELEASE_RESOURCES,
        )
        .unwrap();
    // In order to execute our operation, we have to build a command buffer.
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        // The command buffer only does one thing: execute the compute pipeline. This is called a
        // *dispatch* operation.
        //
        // Note that we clone the pipeline and the set. Since they are both wrapped in an `Arc`,
        // this only clones the `Arc` and not the whole pipeline or set (which aren't cloneable
        // anyway). In this example we would avoid cloning them since this is the last time we use
        // them, but in real code you would probably need to clone them.
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap()
        .dispatch([1, 1, 1])
        .unwrap();
    println!(
        "enable subgroup size control {}",
        device.enabled_extensions().ext_subgroup_size_control
    );

    // Finish building the command buffer by calling `build`.
    let command_buffer = builder.build().unwrap();

    // Let's execute this command buffer now.
    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        // This line instructs the GPU to signal a *fence* once the command buffer has finished
        // execution. A fence is a Vulkan object that allows the CPU to know when the GPU has
        // reached a certain point. We need to signal a fence here because below we want to block
        // the CPU until the GPU has reached that point in the execution.
        .then_signal_fence_and_flush()
        .unwrap();

    // Blocks execution until the GPU has finished the operation. This method only exists on the
    // future that corresponds to a signalled fence. In other words, this method wouldn't be
    // available if we didn't call `.then_signal_fence_and_flush()` earlier. The `None` parameter
    // is an optional timeout.
    //
    // Note however that dropping the `future` variable (with `drop(future)` for example) would
    // block execution as well, and this would be the case even if we didn't call
    // `.then_signal_fence_and_flush()`. Therefore the actual point of calling
    // `.then_signal_fence_and_flush()` and `.wait()` is to make things more explicit. In the
    // future, if the Rust language gets linear types vulkano may get modified so that only
    // fence-signalled futures can get destroyed like this.
    future.wait(None).unwrap();

    // Now that the GPU is done, the content of the buffer should have been modified. Let's check
    // it out. The call to `read()` would return an error if the buffer was still in use by the
    // GPU.
    let _data_buffer_content = b_sort_buffer.read().unwrap();
    let b_global_hist = b_globalhist_buffer.read().unwrap();
    let b_alt_buffer_content = b_alt_buffer.read().unwrap();
    let out_reduction_content = out_reduction_buffer.read().unwrap();
    /*
    let mut file = match File::create("output.txt") {
        Err(why) => panic!("couldn't create: {}", why),
        Ok(file) => file,
    };

    for data in _data_buffer_content.iter(){
        file.write_u32::<LittleEndian>(*data).unwrap();
    }
    */

    for n in 0..15360 / 2 {
        println!("sorted[{}]: {}", n, _data_buffer_content[n as usize]);
    }
    for n in 0..15360 / 2 {
        println!("b_alt[{}]: {}", n, b_alt_buffer_content[n as usize]);
    }

    for n in 0..1024 {
        println!("b_globalHist[{}]: {}", n, b_global_hist[n as usize]);
    }
    for n in 0..256 {
        println!(
            "out_reduction_buffer[{}]: {}",
            n, out_reduction_content[n as usize]
        );
    }

    println!("Success");
}
