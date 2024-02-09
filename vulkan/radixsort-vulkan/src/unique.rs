use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::num_traits::ToPrimitive;
use cgmath::Vector4;
use rand::Rng;
use std::sync::Arc;
use std::{fs::File, io::Write};
use vulkano::buffer::BufferContents;
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

use std::time::{Duration, Instant};

pub fn unique(u_sort: &mut Vec<u32>, n: u32) -> u32 {
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
                subgroup_size_control: true,
                ..Features::empty()
            },
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shader/unique.comp",
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
    let u_sort_buffer = Buffer::from_iter(
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
        u_sort.clone(),
    )
    .unwrap();

    let n_buffer = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        n,
    )
    .unwrap();

    let n_unique_buffer = Buffer::new_sized::<u32>(
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
            WriteDescriptorSet::buffer(0, u_sort_buffer.clone()),
            WriteDescriptorSet::buffer(1, n_buffer.clone()),
            WriteDescriptorSet::buffer(2, n_unique_buffer.clone()),
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

    // Finish building the command buffer by calling `build`.
    let command_buffer = builder.build().unwrap();

    let start = Instant::now();
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
    let duration = start.elapsed();
    println!("Time elapsed in unique() is: {:?}", duration); 

    let u_sort_content = u_sort_buffer.read().unwrap();
    let n_unique_content = n_unique_buffer.read().unwrap();
    for (i, val) in u_sort_content.iter().enumerate() {
        u_sort[i] = *val;
    }
    return n_unique_content.to_u32().unwrap();
}
