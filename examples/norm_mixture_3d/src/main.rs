#[macro_use]
extern crate clap;

use clap::{App, Arg};
use rand::distributions::{Distribution};
use statrs::distribution::{Uniform, Normal};
use nalgebra::{Vector3, Matrix3};

use std::time::Instant;


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
            .default_value("100"),
    )
    .arg(
        Arg::with_name("nb_test_samples")
            .help("Number of testing samples")
            .short("v")
            .takes_value(true)
            .default_value("100"),
    )
    .arg(
        Arg::with_name("nb_components")
            .help("Number of Gaussian component")
            .short("k")
            .takes_value(true)
            .default_value("2"),
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
let reg_cov = value_t_or_exit!(matches.value_of("reg_cov"), f32);
let nb_iter = value_t_or_exit!(matches.value_of("nb_iter"), usize);
let epsilon = value_t_or_exit!(matches.value_of("epsilon"), f32);


    // specify mean and covariance of our multi-variate normal

    
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0., 1.).unwrap();
    let uniform = Uniform::new(0., 1.).unwrap();


    let pi = vec![1 as f32, 2 as f32];
    let total = pi.iter()
        .sum::<f32>();
    let pi = pi
        .iter()
        .map(|p| p/total)
        .collect::<Vec<_>>();


    let means = vec![
        Vector3::from_vec(vec![-1.0f32, 0.0f32, 0.0f32]), 
        Vector3::from_vec(vec![1.0f32, 0.0f32, 0.0f32])];
    let covs = vec![
        Matrix3::from_vec(vec![0.5f32, 0.0f32, 0.0f32,
            0.0f32, 0.5f32, 0.0f32,
            0.0f32, 0.0f32, 0.5f32]), 
        Matrix3::from_vec(vec![0.5f32, 0.0f32, 0.0f32,
            0.0f32, 0.5f32, 0.0f32,
            0.0f32, 0.0f32, 0.5f32])];
    let covs_chol = covs.iter()
        .map(|&cov| cov.clone().cholesky().unwrap().l())
        .collect::<Vec<_>>();

    let u_train = uniform
        .sample_iter(&mut rng)
        .take(nb_train_samples)
        .map(|u| u as f32)
        .collect::<Vec<_>>();

    let u_norm_train = (0..nb_train_samples)
        .map(|_| Vector3::from_vec(normal.sample_iter(&mut rng)
            .take(3)
            .map(|u| u as f32)
            .collect::<Vec<_>>()))
        .collect::<Vec<_>>();

    let x_train = u_train.iter()
        .map(|&u| (u as f32) < pi[1])
        .zip(u_norm_train.iter())
        .map(|(boolean, &u_norm)| means[boolean as usize] + covs_chol[boolean as usize]*u_norm)
        .collect::<Vec<_>>();

    let z_train = u_train.iter()
        .map(|&u| (u as f32) <  pi[1])
        .map(|boolean|  boolean as usize)
        .collect::<Vec<_>>();

    let mut my_gmm = gmm::init_gmm3_d(nb_train_samples, 
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
        .sum::<usize>() as f32 / nb_train_samples as f32;
    if train_accuracy < 0.25 {
        train_accuracy = 1 as f32 - train_accuracy // index of gaussian shifted
    }
    println!("Training accuracy: {}", train_accuracy);

    let u_test = uniform
        .sample_iter(&mut rng)
        .take(nb_test_samples)
        .map(|u| u as f32)
        .collect::<Vec<_>>();

    let u_norm_test = (0..nb_test_samples)
        .map(|_| Vector3::<f32>::from_vec(normal.sample_iter(&mut rng)
            .take(3)
            .map(|u| u as f32)
            .collect::<Vec<_>>()))
        .collect::<Vec<_>>();

    // println!("{:#?}",u_norm_test);
    let x_test = u_test.iter()
        .map(|&u| (u as f32) < pi[1])
        .zip(u_norm_test.iter())
        .map(|(boolean, &u_norm)| means[boolean as usize] + covs_chol[boolean as usize]*u_norm)
        .collect::<Vec<_>>();

    // println!("{:#?}",x_test);
    let z_test = u_test.iter()
        .map(|&u| (u as f32) <  pi[1])
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
        .sum::<usize>() as f32 / nb_test_samples as f32;
    if test_accuracy < 0.25 {
        test_accuracy = 1 as f32 - test_accuracy // index of gaussian shifted
    }
    println!("Testing accuracy: {}", test_accuracy);
    
}
