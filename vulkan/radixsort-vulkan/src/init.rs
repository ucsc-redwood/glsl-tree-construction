use rand::Rng;

fn generate_random_number(min: f32, range: f32) -> f32 {
    let mut rng = rand::thread_rng(); // Get a random number generator
    let rand_float: f32 = rng.gen(); // Generate a random float between 0.0 and 1.0

    rand_float * range + min // Scale and shift the random number to the desired range
}

pub fn init_random(
    u_data: &mut Vec<[f32; 4]>, /* ,
                                n : i64,
                                min : f32,
                                range : f32*/
) {
    // srand(seed);

    let min = 0.0;
    let max = 1024.0;
    let range = max - min;
    let mut rng = rand::thread_rng();

    //unsigned int my_seed = seed + tid;

    for i in 0..15360 {
        u_data[i][0] = generate_random_number(min, range);
        u_data[i][1] = generate_random_number(min, range);
        u_data[i][2] = generate_random_number(min, range);
        u_data[i][3] = 1.0;
    }
}
