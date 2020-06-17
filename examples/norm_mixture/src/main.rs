#[macro_use]
extern crate clap;

use clap::{App, Arg};
use rand::distributions::{Distribution};
use statrs::distribution::{Uniform, Normal};
use std::time::{Instant};

use gmm::EM;


fn main() {

    let matches = App::new("Gaussian Mixture Model")
        .version("0.1")
        .about("Mixture of normal density")
        .arg(
            Arg::with_name("nb_train_samples")
                .help("Number of training samples")
                .short("t")
                .takes_value(true)
                .default_value("100000"),
        )
        .arg(
            Arg::with_name("nb_test_samples")
                .help("Number of testing samples")
                .short("v")
                .takes_value(true)
                .default_value("100000"),
        )
        .arg(
            Arg::with_name("nb_components")
                .help("Number of Gaussian component")
                .short("k")
                .takes_value(true)
                .default_value("10"),
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
        .get_matches();

    // Hyperparameters
    let nb_train_samples = value_t_or_exit!(matches.value_of("nb_train_samples"), usize);
    let nb_test_samples = value_t_or_exit!(matches.value_of("nb_test_samples"), usize);
    let nb_components = value_t_or_exit!(matches.value_of("nb_components"), usize);
    let reg_cov = value_t_or_exit!(matches.value_of("reg_cov"), f64);
    let nb_iter = value_t_or_exit!(matches.value_of("nb_iter"), usize);
    let epsilon = value_t_or_exit!(matches.value_of("epsilon"), f64);

    // Data generation
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(0 as f64, 1 as f64).unwrap();
    
    let pi = vec![1 as f64, 2 as f64];
    let total = pi.iter()
        .sum::<f64>();
    let pi = pi
        .iter()
        .map(|p| p/total)
        .collect::<Vec<_>>();
    let means = vec![-0.25 as f64, 0.35 as f64];
    let covs = vec![0.1 as f64, 0.3 as f64];

    let gaussians = means.iter()
    .zip(covs.iter())
    .map(|(mean, cov)| Normal::new(*mean, *cov).unwrap())
    .collect::<Vec<_>>();

    let u_train = uniform
        .sample_iter(&mut rng)
        .take(nb_train_samples)
        .collect::<Vec<_>>();

    let x_train = u_train.iter()
        .map(|u| *u < pi[1])
        .map(|boolean| gaussians[boolean as usize].sample(&mut rng))
        .collect::<Vec<_>>();

    let z_train = u_train.iter()
        .map(|u| *u <  pi[1])
        .map(|boolean|  boolean as usize)
        .collect::<Vec<_>>();


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
    let mut train_accuracy =  my_gmm.z
        .iter()
        .zip(z_train.iter())
        .map(|(z_gmm, z_true)| (z_gmm == z_true) as usize )
        .sum::<usize>() as f64 / nb_train_samples as f64;
    if train_accuracy < 0.25 {
        train_accuracy = 1 as f64 - train_accuracy // index of gaussian shifted
    }
    println!("Training accuracy: {}", train_accuracy);


    let u_test = uniform
        .sample_iter(&mut rng)
        .take(nb_test_samples)
        .collect::<Vec<_>>();

    let x_test = u_test.iter()
        .map(|u| *u < pi[1])
        .map(|boolean| gaussians[boolean as usize].sample(&mut rng))
        .collect::<Vec<_>>();

    let z_test = u_test.iter()
        .map(|u| *u <  pi[1])
        .map(|boolean|  boolean as usize)
        .collect::<Vec<_>>();

    let start = Instant::now();
    let z_predict = my_gmm.predict(&x_test);
    let elapsed = start.elapsed().as_millis();
    println!("time taken to test the gmm : {} ms", elapsed);
    let mut test_accuracy =  z_predict
        .iter()
        .zip(z_test.iter())
        .map(|(z_gmm, z_true)| (z_gmm == z_true) as usize )
        .sum::<usize>() as f64 / nb_test_samples as f64;
    if test_accuracy < 0.25 {
        test_accuracy = 1 as f64 - test_accuracy // index of gaussian shifted
    }
    println!("Testing accuracy: {}", test_accuracy);
}