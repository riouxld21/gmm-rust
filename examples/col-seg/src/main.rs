#[macro_use]
extern crate clap;

use clap::{App, Arg};
use rand::seq::SliceRandom; 
use std::time::Instant;
// use image::{GenericImageView, Pixel};

use gmm::EM;

fn main() {

    let matches = App::new("Gaussian Mixture Model")
        .version("0.1")
        .about("Image segmentation")
        .arg(
            Arg::with_name("input")
                .help("input image to optimize [required]")
                .short("i")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .help("output final image [required]")
                .short("o")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("percent_samples")
                .help("Percentage of image samples used to train")
                .short("p")
                .takes_value(true)
                .default_value("10"),
        )
        .arg(
            Arg::with_name("nb_components")
                .help("Number of Gaussian component")
                .short("k")
                .takes_value(true)
                .default_value("8"),
        )
        .arg(
            Arg::with_name("reg_cov")
                .help("Covariance regulation term ")
                .short("r")
                .takes_value(true)
                .default_value("0.03"),
        )
        .arg(
            Arg::with_name("nb_iter")
                .help("Number of iterations")
                .short("n")
                .takes_value(true)
                .default_value("100"),
        )
        .arg(
            Arg::with_name("epsilon")
                .help("Tolerance at which we stop the EM procedure")
                .short("e")
                .takes_value(true)
                .default_value("0.001"),
        )
        .arg(
            Arg::with_name("color")
                .help("Generate color image")
                .short("c"),
        )
        .get_matches();

    // let with_color = matches.is_present("color");


    // Generate the main object & Register brushes
    let filename_out = value_t_or_exit!(matches.value_of("output"), String);
    let filename = value_t_or_exit!(matches.value_of("input"), String);
    let img = image::open(&filename).expect(&format!("Impossible to open {}", filename));
    let img = img.to_luma();
    let (width, height) = img.dimensions();
    let img_vec = img.into_raw();
    

    // Hyperparameters
    let percent_samples = value_t_or_exit!(matches.value_of("percent_samples"), usize);
    let nb_train_samples = ((width*height) as f64 * percent_samples as f64 / 100 as f64) as usize;
    let nb_components = value_t_or_exit!(matches.value_of("nb_components"), usize);
    let reg_cov = value_t_or_exit!(matches.value_of("reg_cov"), f64);
    let nb_iter = value_t_or_exit!(matches.value_of("nb_iter"), usize);
    let epsilon = value_t_or_exit!(matches.value_of("epsilon"), f64);

    
    let x_train = img_vec
        .choose_multiple(&mut rand::thread_rng(), nb_train_samples)
        .map(|&pixel| pixel as f64 / 255 as f64)
        .collect::<Vec<f64>>();

    
    
    let mut my_gmm = gmm::init_gmm1_d(nb_train_samples, 
        nb_components, 
        reg_cov, 
        nb_iter,
        epsilon,
        &x_train);


    let start = Instant::now();
    my_gmm.run();
    let elapsed = start.elapsed().as_millis();
    println!("time taken to train the gmm: {} ms", elapsed);

    let x_test = img_vec.iter()
        .map(|&pixel| pixel as f64 / 255 as f64)
        .collect::<Vec<f64>>();

    let start = Instant::now();
    let z_predict = my_gmm.predict(&x_test);
    let elapsed = start.elapsed().as_millis();
    println!("time taken to test the gmm: {} ms", elapsed);
    
    let new_img_vec = z_predict
        .iter()
        .map(|&idx| (my_gmm.means[idx] * 255 as f64) as u8)
        .collect::<Vec<u8>>();

    let new_img = image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(width, height, new_img_vec).unwrap();
    new_img
        .save(&filename_out)
        .expect(&format!("Impossible to save {}", filename_out));


}