use rand::Rng;

fn generate_random_number(min: f32, range: f32) -> f32 {
    let mut rng = rand::thread_rng(); // Get a random number generator
    let rand_float: f32 = rng.gen(); // Generate a random float between 0.0 and 1.0

    rand_float * range + min // Scale and shift the random number to the desired range
}

pub fn init_random(u_data: &mut Vec<[f32; 4]>, n: u32, min: f32, range: f32) {
    // srand(seed);

    let mut rng = rand::thread_rng();

    //unsigned int my_seed = seed + tid;

    for i in 0..n {
        u_data[i as usize][0] = generate_random_number(min, range);
        u_data[i as usize][1] = generate_random_number(min, range);
        u_data[i as usize][2] = generate_random_number(min, range);
        u_data[i as usize][3] = 1.0;
    }
}


pub fn init_radixsort(u_data: &mut Vec<[f32; 4]>, n: u32) {
    for i in 0..n {
        u_data[i as usize][0] = i as f32;
        u_data[i as usize][1] = i as f32;
        u_data[i as usize][2] = i as f32;
        u_data[i as usize][3] = 1.0;
    }
}
