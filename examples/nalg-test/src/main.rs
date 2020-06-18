#[macro_use]
use nalgebra::{Vector3, Matrix3};

use std::time::Instant;


use gmm::EM;

fn main() {

    let x1 = Vector3::from_vec(vec![0.0, 0.0, 0.0]);
    let v1 = Vector3::from_vec(vec![1.0, 0.0, 0.0]);
    let m1 = Matrix3::from_vec(vec![2., 0.0, 0.0,
            0.0, 2., 0.0,
            0.0, 0.0,2.]);


    
    let start = Instant::now();
    let LU = m1.clone().cholesky().unwrap();
    let elapsed = start.elapsed().as_nanos();
    println!("time taken det1: {} ns", elapsed);

    let start = Instant::now();
    let det = LU.l().determinant();
    let elapsed = start.elapsed().as_nanos();
    println!("time taken det: {} ns, result: {}", elapsed, det);

    let start = Instant::now();
    let inv =  LU.inverse();
    let elapsed = start.elapsed().as_nanos();
    println!("time taken inv: {} ns, result: {:#?}", elapsed, inv);


    let start = Instant::now();
    let mut mvn = gmm::init_mvn(3, &v1, &m1);
    let elapsed = start.elapsed().as_nanos();
    println!("time taken build mvn: {} ns", elapsed);


    let start = Instant::now();
    let pdf = mvn.pdf(&x1);
    let elapsed = start.elapsed().as_nanos();
    println!("time taken pdf: {} ns, result: {}", elapsed, pdf);
}