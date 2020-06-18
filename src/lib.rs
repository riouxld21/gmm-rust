
use nalgebra::{VectorN, MatrixN, DimName, DefaultAllocator, Dim, allocator::Allocator};

use rand::distributions::{Distribution};
use statrs::distribution::{Uniform, Normal, Continuous};
// use rayon::prelude::*;
use std::cmp::Ordering;

pub trait EM<T> {
    fn initialization(&mut self) -> ();
    fn normalization(&mut self) -> ();
    fn expectation(&mut self) -> ();
    fn maximization(&mut self) -> ();
    fn log_likelihood(&mut self) -> ();
    fn run(&mut self) -> ();
    fn predict(&mut self, test_samples: & Vec<T>) -> Vec<usize>;
}


pub struct GMM1D<'a> {
    nb_samples: usize,
    nb_components: usize,
    cov_reg: f32,
    nb_iter: usize,
    epsilon: f32,
    samples: &'a Vec<f32>,
    pub pi: Vec<f32>,
    pub means: Vec<f32>,
    pub covs: Vec<f32>,
    pub gaussians: Vec<Normal>,
    pub z: Vec<usize>,
    weights: Vec<f32>,
    gamma_norm: Vec<f32>,
    gamma: Vec<Vec<f32>>,
    pub log_likelihoods: Vec<f32>,
}


pub fn init_gmm1_d( nb_samples: usize,
    nb_components: usize,
    cov_reg: f32,
    nb_iter:usize,
    epsilon:f32,
    samples: &Vec<f32>) -> GMM1D{
        return GMM1D {
            nb_samples: nb_samples,
            nb_components: nb_components,
            cov_reg: cov_reg,
            nb_iter: nb_iter,
            epsilon: epsilon,
            samples: &samples,
            pi: Vec::<f32>::new(),
            means: Vec::<f32>::new(),
            covs: Vec::<f32>::new(),
            gaussians: Vec::<Normal>::new(),
            z: Vec::<usize>::new(),
            weights: Vec::<f32>::new(),
            gamma_norm: Vec::<f32>::new(),
            gamma: Vec::<Vec::<f32>>::new(),
            log_likelihoods: Vec::<f32>::new(),
        }
    }

impl<'a> EM<f32> for GMM1D<'a> {

    fn initialization(&mut self) {

        let mut rng = rand::thread_rng();
        let uniform_covs = Uniform::new(0.05, 0.5).unwrap();
        let uniform_means = Uniform::new(-0.5, 0.5).unwrap();

        self.weights = vec![0 as f32; self.nb_components];
        self.gamma_norm = vec![0 as f32; self.nb_samples];
        self.gamma = vec![vec![0 as f32; self.nb_components]; self.nb_samples];
        self.z = vec![0 as usize; self.nb_samples];

        self.pi = (0..self.nb_components)
            .map(|_| 1 as f32 / self.nb_components as f32)
            .collect::<Vec<_>>();
        self.means = uniform_means.sample_iter(&mut rng)
            .take(self.nb_components)
            .map(|u| u as f32)
            .collect::<Vec<_>>();
        self.covs = uniform_covs.sample_iter(&mut rng)
            .take(self.nb_components)
            .map(|u| u as f32)
            .collect::<Vec<_>>();
        self.gaussians = self.means.iter()
            .zip(self.covs.iter())
            .map(|(&mean, &cov)| Normal::new(mean as f64, cov as f64).unwrap())
            .collect::<Vec<_>>();

    }

    fn normalization(&mut self) {
        // for i in 0..self.nb_samples {
        //     self.gamma_norm[i] = 0 as f32;
        //     for j in 0..self.nb_components {
        //         self.gamma_norm[i] += self.pi[j]*self.gaussians[j].pdf(self.samples[i]);
        //     }
        // }
        
        self.gamma_norm = self.samples
            .iter()
            .map(|&x| self.gaussians
                .iter()
                .zip(self.pi.iter())
                .map(|(gaussian, &p)| p * gaussian.pdf(x as f64) as f32)
                .sum()
            ).collect::<Vec<f32>>();
    }

    fn expectation(&mut self) {
        // for i in 0..self.nb_samples {
        //     for j in 0..self.nb_components {
        //         self.gamma[i][j] = self.pi[j]*self.gaussians[j].pdf(self.samples[i])/self.gamma_norm[i];
        //     }
        // }

        
        self.gamma = self.samples
            .iter()
            .zip(self.gamma_norm.iter())
            .map(|(&x, cst)| self.gaussians
                .iter()
                .zip(self.pi.iter())
                .map(|(gaussian, &p)| p * gaussian.pdf(x as f64) as f32/cst)
                .collect::<Vec<f32>>()
            ).collect::<Vec<_>>();
    }

    fn maximization(&mut self) {
        
        // for i in 0..self.nb_samples {
        //     let mut max_value = std::f32::NEG_INFINITY;
        //     let mut max_idx = 0;
        //     for j in 0..self.nb_components {
        //         if self.gamma[i][j] > max_value {
        //             max_value = self.gamma[i][j];
        //             max_idx = j;
        //         }
        //     }
        //     self.z[i] = max_idx;
        // }

        self.z = self.gamma
            .iter()
            .map(|gamma| gamma
                .iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap()
            )
            .collect::<Vec<usize>>();

        for j in 0..self.nb_components {
            self.weights[j] = 0 as f32;
            for i in 0..self.nb_samples {
                self.weights[j] += self.gamma[i][j];
            }
        }


        self.pi = self.weights
            .iter()
            .map(|x| x / self.nb_samples as f32)
            .collect::<Vec<f32>>();

        for j in 0..self.nb_components {
            // self.pi[j] = self.weights[j]/self.nb_samples as f32;

            self.means[j] = 0 as f32;
            for i in 0..self.nb_samples {
                self.means[j] += self.gamma[i][j]*self.samples[i];
            }
            self.means[j] /= self.weights[j];
    
    
            self.covs[j] = 0 as f32;
            for i in 0..self.nb_samples {
                self.covs[j] += self.gamma[i][j]*(self.samples[i]-self.means[j])*(self.samples[i]-self.means[j]);
            }
            self.covs[j] /= self.weights[j];
            self.covs[j] += self.cov_reg;

            // self.gaussians[j] = Normal::new(self.means[j], self.covs[j]).unwrap();
        }


        self.gaussians = self.means.iter()
            .zip(self.covs.iter())
            .map(|(&mean, &cov)| Normal::new(mean as f64, cov as f64).unwrap())
            .collect::<Vec<_>>();
    }

    fn log_likelihood(&mut self) {

        // let mut sum_log = 0 as f32;
        // for i in 0..self.nb_samples {
        //     sum_log += self.gamma_norm[i].ln();
        // }

        let sum_log = self.gamma_norm
            .iter()
            .map(|gamma_norm| gamma_norm.ln())
            .sum::<f32>();

        self.log_likelihoods.push(sum_log);
    }

    fn run(&mut self) {
        self.initialization();
        self.normalization();
        self.log_likelihood();
        for tok in 0..self.nb_iter {
            self.expectation();
            self.maximization();
            self.normalization();
            let temp = self.log_likelihoods.last().copied().unwrap();
            self.log_likelihood();
            let error = temp - self.log_likelihoods.last().copied().unwrap();
            let rel_error = 2.*error / (temp + self.log_likelihoods.last().copied().unwrap());
            println!("Iteration {:4} -- rel_error {:.6} ", tok, rel_error);
            
            if rel_error.abs() < self.epsilon as f32 {
                break;
            }
        }
        // println!("pi {:#?}", self.pi);
        // println!("gaussians {:#?}", self.gaussians);
    }

    fn predict(&mut self, test_samples: &Vec<f32>) -> Vec<usize> {

        let test_gamma_norm = test_samples
            .iter()
            .map(|&x| self.gaussians
                .iter()
                .zip(self.pi.iter())
                .map(|(gaussian, &p)| p * gaussian.pdf(x as f64) as f32)
                .sum()
            ).collect::<Vec<f32>>();
            
        let test_gamma = test_samples
            .iter()
            .zip(test_gamma_norm.iter())
            .map(|(&x, cst)| self.gaussians
                .iter()
                .zip(self.pi.iter())
                .map(|(gaussian, &p)| p * gaussian.pdf(x as f64) as f32/cst)
                .collect::<Vec<f32>>()
            ).collect::<Vec<_>>();

        let test_z = test_gamma
            .iter()
            .map(|gamma| gamma .iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap()
            )
            .collect::<Vec<usize>>();

        return test_z
    }
}

#[allow(dead_code)]
#[derive(Debug,Clone)]
pub struct MVN<D>  where
        
D: Dim + DimName + nalgebra::DimMin<D, Output = D>  + nalgebra::DimSub<nalgebra::Dynamic>,
        DefaultAllocator: Allocator<f32, D>,
        DefaultAllocator: Allocator<f32, D, D>,
        DefaultAllocator: Allocator<f32, nalgebra::U1, D>,
        DefaultAllocator: Allocator<(usize, usize), <D as nalgebra::DimMin<D>>::Output>, 
        {
    dim: usize,
    mean: VectorN<f32, D>,
    cov: MatrixN<f32, D>,
    det_sqrt: f32,
    inv: MatrixN<f32, D>,
    cst: f32,
}

impl<D> MVN<D> where
    D: Dim + DimName + nalgebra::DimMin<D, Output = D> + nalgebra::DimSub<nalgebra::Dynamic>,
    DefaultAllocator: Allocator<f32, D>,
    DefaultAllocator: Allocator<f32, D, D>,
    DefaultAllocator: Allocator<f32, nalgebra::U1, D>,
    DefaultAllocator: Allocator<(usize, usize), <D as nalgebra::DimMin<D>>::Output>, 
{
    pub fn pdf(&self, x: &VectorN<f32,D>) -> f32 { 
        let temp = ((x-self.mean.clone()).transpose()*self.inv.clone()*(x-self.mean.clone()))[0];
        return self.cst*(-0.5 as f32 * temp).exp()
    }
}

pub fn init_mvn<D>(mean: VectorN<f32, D>, cov: MatrixN<f32, D>) -> MVN<D> where
    D: Dim + DimName + nalgebra::DimMin<D, Output = D>  + nalgebra::DimSub<nalgebra::Dynamic>,
    DefaultAllocator: Allocator<f32, D>,
    DefaultAllocator: Allocator<f32, D, D>,
    DefaultAllocator: Allocator<f32, nalgebra::U1, D>,
    DefaultAllocator: Allocator<(usize, usize), <D as nalgebra::DimMin<D>>::Output>,
{
    let n = cov.nrows();
    let chol = cov.clone().cholesky().unwrap();
    let det_sqrt = chol.l().determinant();
    let cst = 1./((2.* std::f32::consts::PI).powf(n as f32 / 2.) * chol.l().determinant());

    return MVN {
        dim: n,
        mean: mean,
        cov: cov,
        det_sqrt: det_sqrt,
        inv: chol.inverse(),
        cst: cst,
    }
}





pub struct GMM3D<'a, D> where
D: Dim + DimName + nalgebra::DimMin<D, Output = D>  + nalgebra::DimSub<nalgebra::Dynamic>,
DefaultAllocator: Allocator<f32, D>,
DefaultAllocator: Allocator<f32, D, D>,
DefaultAllocator: Allocator<f32, nalgebra::U1, D>,
DefaultAllocator: Allocator<(usize, usize), <D as nalgebra::DimMin<D>>::Output>,
{
    nb_dim: usize,
    nb_samples: usize,
    nb_components: usize,
    nb_iter: usize,
    epsilon: f32,
    samples: &'a Vec<VectorN<f32, D>>,
    pub pi: Vec<f32>,
    pub means: Vec<VectorN<f32, D>>,
    pub covs: Vec<MatrixN<f32, D>>,
    pub cov_reg: MatrixN<f32, D>,
    pub mvns: Vec<MVN<D>>,
    pub z: Vec<usize>,
    weights: Vec<f32>,
    gamma_norm: Vec<f32>,
    gamma: Vec<Vec<f32>>,
    pub log_likelihoods: Vec<f32>,
}


pub fn init_gmm3_d<'a, D>( nb_samples: usize,
    nb_components: usize,
    cov_reg: f32,
    nb_iter:usize,
    epsilon:f32,
    samples: &'a Vec<VectorN<f32, D>>) -> GMM3D<'a, D>
    where
        D: Dim + DimName + nalgebra::DimMin<D, Output = D>  + nalgebra::DimSub<nalgebra::Dynamic>,
        DefaultAllocator: Allocator<f32, D>,
        DefaultAllocator: Allocator<f32, D, D>,
        DefaultAllocator: Allocator<f32, nalgebra::U1, D>,
        DefaultAllocator: Allocator<(usize, usize), <D as nalgebra::DimMin<D>>::Output>,
{
    let nb_dim = samples[0].nrows();   
    return GMM3D {
            nb_dim: nb_dim,
            nb_samples: nb_samples,
            nb_components: nb_components,
            nb_iter: nb_iter,
            epsilon: epsilon,
            samples: &samples,
            pi: Vec::<f32>::new(),
            means: Vec::<VectorN<f32, D>>::new(),
            covs: Vec::<MatrixN<f32, D>>::new(),
            cov_reg: MatrixN::<f32, D>::from_diagonal_element(cov_reg),
            mvns: Vec::<MVN<D>>::new(),
            z: Vec::<usize>::new(),
            weights: Vec::<f32>::new(),
            gamma_norm: Vec::<f32>::new(),
            gamma: Vec::<Vec::<f32>>::new(),
            log_likelihoods: Vec::<f32>::new(),
        }
    }


impl<'a, D> EM<VectorN<f32, D>> for GMM3D<'a, D> where

D: Dim + DimName + nalgebra::DimMin<D, Output = D>  + nalgebra::DimSub<nalgebra::Dynamic>,
DefaultAllocator: Allocator<f32, D>,
DefaultAllocator: Allocator<f32, D, D>,
DefaultAllocator: Allocator<f32, nalgebra::U1, D>,
DefaultAllocator: Allocator<(usize, usize), <D as nalgebra::DimMin<D>>::Output>,  
{

    fn initialization(&mut self) {

        let mut rng = rand::thread_rng();
        // let uniform_covs = Uniform::new(0.05_f32, 0.5_f32).unwrap();
        let uniform_means = Uniform::new(-1., 1.).unwrap();

        self.weights = vec![0 as f32; self.nb_components];
        self.gamma_norm = vec![0 as f32; self.nb_samples];
        self.gamma = vec![vec![0 as f32; self.nb_components]; self.nb_samples];
        self.z = vec![0 as usize; self.nb_samples];


        self.pi = (0..self.nb_components)
            .map(|_| 1 as f32 / self.nb_components as f32)
            .collect::<Vec<_>>();
        
        self.means = (0..self.nb_components)
            .map(|_| VectorN::<f32, D>::from_vec(uniform_means.sample_iter(&mut rng)
                .take(self.nb_dim)
                .map(|u| u as f32)
                .collect::<Vec<_>>()))
            .collect::<Vec<_>>();

        self.covs = (0..self.nb_components)
            .map(|_| MatrixN::<f32, D>::from_diagonal_element(0.05))
        .collect::<Vec<_>>();

        self.mvns = self.means
            .iter()
            .zip(self.covs.iter())
            .map(|(mean, cov)| init_mvn::<D>(mean.clone(), cov.clone()))
            .collect::<Vec<_>>();

        // println!("pi {:#?}", self.pi);
        // println!("means {:#?}", self.means);
        // println!("covs {:#?}", self.covs);

    }

    fn normalization(&mut self) {
        // for i in 0..self.nb_samples {
        //     self.gamma_norm[i] = 0 as f32;
        //     for j in 0..self.nb_components {
        //         let mvn = MultivariateNormal::from_mean_and_covariance(&self.means[j], &self.covs[j]).unwrap();
        //         self.gamma_norm[i] += self.pi[j] * mvn.pdf(&(self.samples[i]).transpose())[0];
        //     }
        // }
        
        
        self.gamma_norm = self.samples
            .iter()
            .map(|x| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)| p * mvn.pdf(&x))
                .sum()
            ).collect::<Vec<f32>>();
    }

    fn expectation(&mut self) {
        
        // for i in 0..self.nb_samples {
        //     for j in 0..self.nb_components {
        //         let mvn = MultivariateNormal::from_mean_and_covariance(&self.means[j], &self.covs[j]).unwrap();
        //         self.gamma[i][j] = self.pi[j]*mvn.pdf(&(self.samples[i]).transpose())[0]/self.gamma_norm[i];
        //     }
        // }

        self.gamma = self.samples
            .iter()
            .zip(self.gamma_norm.iter())
            .map(|(x, cst)| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)| p * mvn.pdf(x)/cst)
                .collect::<Vec<f32>>()
            ).collect::<Vec<Vec<f32>>>();
    }

    fn maximization(&mut self) {
        
        // for i in 0..self.nb_samples {
        //     let mut max_value = std::f32::NEG_INFINITY;
        //     let mut max_idx = 0;
        //     for j in 0..self.nb_components {
        //         if self.gamma[i][j] > max_value {
        //             max_value = self.gamma[i][j];
        //             max_idx = j;
        //         }
        //     }
        //     self.z[i] = max_idx;
        // }

        self.z = self.gamma
            .iter()
            .map(|gamma| gamma
                .iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap()
            )
            .collect::<Vec<usize>>();

        for j in 0..self.nb_components {
            self.weights[j] = 0 as f32;
            for i in 0..self.nb_samples {
                self.weights[j] += self.gamma[i][j];
            }
        }

        self.pi = self.weights
            .iter()
            .map(|x| x / self.nb_samples as f32)
            .collect::<Vec<f32>>();

        for j in 0..self.nb_components {
            // self.pi[j] = self.weights[j]/self.nb_samples as f32;

            self.means[j] = VectorN::<f32, D>::from_vec(vec![0.0f32; self.nb_dim]);
            for i in 0..self.nb_samples {
                self.means[j] += self.gamma[i][j]*self.samples[i].clone();
            }
            self.means[j] /= self.weights[j];
    
    
            self.covs[j] = MatrixN::<f32, D>::from_vec(vec![0.0f32; self.nb_dim*self.nb_dim]);
            for i in 0..self.nb_samples {
                self.covs[j] += self.gamma[i][j]*(self.samples[i].clone()-self.means[j].clone())*(self.samples[i].clone()-self.means[j].clone()).transpose();
            }
            self.covs[j] /= self.weights[j];
            self.covs[j] += self.cov_reg.clone();

            self.mvns[j] = init_mvn(self.means[j].clone(), self.covs[j].clone());
        }


        // self.gaussians = self.means.iter()
        //     .zip(self.covs.iter())
        //     .map(|(&mean, &cov)| Normal::new(mean, cov).unwrap())
        //     .collect::<Vec<_>>();
    }

    fn log_likelihood(&mut self) {

        // // let mut sum_log = 0 as f32;
        // // for i in 0..self.nb_samples {
        // //     sum_log += self.gamma_norm[i].ln();
        // // }

        let sum_log = self.gamma_norm
            .iter()
            .map(|gamma_norm| gamma_norm.ln())
            .sum::<f32>();

        self.log_likelihoods.push(sum_log);
    }

    fn run(&mut self) {
        self.initialization();
        self.normalization();
        self.log_likelihood();
        for tok in 0..self.nb_iter {
            self.expectation();
            self.maximization();
            self.normalization();
            let temp = self.log_likelihoods.last().copied().unwrap();
            self.log_likelihood();
            let error = temp - self.log_likelihoods.last().copied().unwrap();
            let rel_error = 2.*error / (temp + self.log_likelihoods.last().copied().unwrap());
            println!("Iteration {:4} -- rel_error {:.6} ", tok, rel_error);
            
            if rel_error.abs() < self.epsilon as f32 {
                break;
            }
        }
        // println!("pi {:#?}", self.pi);
        // println!("means {:#?}", self.means);
        // println!("covs {:#?}", self.covs);
    }

    fn predict(&mut self, test_samples: &Vec<VectorN<f32, D>>) -> Vec<usize> {
        

        // let nb_samples_test = test_samples.len();

        // let mut test_gamma_norm = vec![0.0; nb_samples_test];
        // for i in 0..nb_samples_test {
        //     test_gamma_norm[i] = 0 as f32;
        //     for j in 0..self.nb_components {
        //         let mvn = MultivariateNormal::from_mean_and_covariance(&self.means[j], &self.covs[j]).unwrap();
        //         test_gamma_norm[i] += self.pi[j] * mvn.pdf(&(test_samples[i]).transpose())[0];
        //     }
        // }

        // let mut test_gamma= vec![vec![0.0; self.nb_components]; nb_samples_test];
        // for i in 0..nb_samples_test {
        //     for j in 0..self.nb_components {
        //         let mvn = MultivariateNormal::from_mean_and_covariance(&self.means[j], &self.covs[j]).unwrap();
        //         test_gamma[i][j] = self.pi[j]*mvn.pdf(&(test_samples[i]).transpose())[0]/test_gamma_norm[i];
        //     }
        // }

        let test_gamma_norm = test_samples
            .iter()
            .map(|x| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)|
                    p * mvn.pdf(&x)
                )
                .sum()
            ).collect::<Vec<f32>>();
            
        let test_gamma = test_samples
            .iter()
            .zip(test_gamma_norm.iter())
            .map(|(x, cst)| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)| p * mvn.pdf(&x)/cst)
                .collect::<Vec<f32>>()
            ).collect::<Vec<Vec<f32>>>();

            
        let test_z = test_gamma
            .iter()
            .map(|gamma| gamma.iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap()
            )
            .collect::<Vec<usize>>();

        return test_z
    }
}