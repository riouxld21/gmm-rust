use nalgebra::{Vector3, Matrix3};
use rand::distributions::{Distribution};
use statrs::distribution::{Uniform, Normal, Continuous};
use rayon::prelude::*;
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
    cov_reg: f64,
    nb_iter: usize,
    epsilon: f64,
    samples: &'a Vec<f64>,
    pub pi: Vec<f64>,
    pub means: Vec<f64>,
    pub covs: Vec<f64>,
    pub gaussians: Vec<Normal>,
    pub z: Vec<usize>,
    weights: Vec<f64>,
    gamma_norm: Vec<f64>,
    gamma: Vec<Vec<f64>>,
    pub log_likelihoods: Vec<f64>,
}


pub fn init_gmm1_d( nb_samples: usize,
    nb_components: usize,
    cov_reg: f64,
    nb_iter:usize,
    epsilon:f64,
    samples: &Vec<f64>) -> GMM1D{
        return GMM1D {
            nb_samples: nb_samples,
            nb_components: nb_components,
            cov_reg: cov_reg,
            nb_iter: nb_iter,
            epsilon: epsilon,
            samples: &samples,
            pi: Vec::<f64>::new(),
            means: Vec::<f64>::new(),
            covs: Vec::<f64>::new(),
            gaussians: Vec::<Normal>::new(),
            z: Vec::<usize>::new(),
            weights: Vec::<f64>::new(),
            gamma_norm: Vec::<f64>::new(),
            gamma: Vec::<Vec::<f64>>::new(),
            log_likelihoods: Vec::<f64>::new(),
        }
    }

impl<'a> EM<f64> for GMM1D<'a> {

    fn initialization(&mut self) {

        let mut rng = rand::thread_rng();
        let uniform_covs = Uniform::new(0.05_f64, 0.5_f64).unwrap();
        let uniform_means = Uniform::new(-0.5_f64, 0.5_f64).unwrap();

        self.weights = vec![0 as f64; self.nb_components];
        self.gamma_norm = vec![0 as f64; self.nb_samples];
        self.gamma = vec![vec![0 as f64; self.nb_components]; self.nb_samples];
        self.z = vec![0 as usize; self.nb_samples];

        self.pi = (0..self.nb_components)
            .map(|_| 1 as f64 / self.nb_components as f64)
            .collect::<Vec<_>>();
        self.means = uniform_means.sample_iter(&mut rng)
            .take(self.nb_components)
            .collect::<Vec<_>>();
        self.covs = uniform_covs.sample_iter(&mut rng)
            .take(self.nb_components)
            .collect::<Vec<_>>();
        self.gaussians = self.means.iter()
            .zip(self.covs.iter())
            .map(|(&mean, &cov)| Normal::new(mean, cov).unwrap())
            .collect::<Vec<_>>();

    }

    fn normalization(&mut self) {
        // for i in 0..self.nb_samples {
        //     self.gamma_norm[i] = 0 as f64;
        //     for j in 0..self.nb_components {
        //         self.gamma_norm[i] += self.pi[j]*self.gaussians[j].pdf(self.samples[i]);
        //     }
        // }
        
        self.gamma_norm = self.samples
            .par_iter()
            .map(|&x| self.gaussians
                .iter()
                .zip(self.pi.iter())
                .map(|(gaussian, &p)| p * gaussian.pdf(x))
                .sum()
            ).collect::<Vec<f64>>();
    }

    fn expectation(&mut self) {
        // for i in 0..self.nb_samples {
        //     for j in 0..self.nb_components {
        //         self.gamma[i][j] = self.pi[j]*self.gaussians[j].pdf(self.samples[i])/self.gamma_norm[i];
        //     }
        // }

        
        self.gamma = self.samples
            .par_iter()
            .zip(self.gamma_norm.par_iter())
            .map(|(&x, cst)| self.gaussians
                .iter()
                .zip(self.pi.iter())
                .map(|(gaussian, &p)| p * gaussian.pdf(x)/cst)
                .collect::<Vec<f64>>()
            ).collect::<Vec<_>>();
    }

    fn maximization(&mut self) {
        
        // for i in 0..self.nb_samples {
        //     let mut max_value = std::f64::NEG_INFINITY;
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
            .par_iter()
            .map(|gamma| gamma
                .iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap()
            )
            .collect::<Vec<usize>>();

        for j in 0..self.nb_components {
            self.weights[j] = 0 as f64;
            for i in 0..self.nb_samples {
                self.weights[j] += self.gamma[i][j];
            }
        }


        self.pi = self.weights
            .iter()
            .map(|x| x / self.nb_samples as f64)
            .collect::<Vec<f64>>();

        for j in 0..self.nb_components {
            // self.pi[j] = self.weights[j]/self.nb_samples as f64;

            self.means[j] = 0 as f64;
            for i in 0..self.nb_samples {
                self.means[j] += self.gamma[i][j]*self.samples[i];
            }
            self.means[j] /= self.weights[j];
    
    
            self.covs[j] = 0 as f64;
            for i in 0..self.nb_samples {
                self.covs[j] += self.gamma[i][j]*(self.samples[i]-self.means[j])*(self.samples[i]-self.means[j]);
            }
            self.covs[j] /= self.weights[j];
            self.covs[j] += self.cov_reg;

            // self.gaussians[j] = Normal::new(self.means[j], self.covs[j]).unwrap();
        }


        self.gaussians = self.means.par_iter()
            .zip(self.covs.par_iter())
            .map(|(&mean, &cov)| Normal::new(mean, cov).unwrap())
            .collect::<Vec<_>>();
    }

    fn log_likelihood(&mut self) {

        // let mut sum_log = 0 as f64;
        // for i in 0..self.nb_samples {
        //     sum_log += self.gamma_norm[i].ln();
        // }

        let sum_log = self.gamma_norm
            .par_iter()
            .map(|gamma_norm| gamma_norm.ln())
            .sum::<f64>();

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
            
            if rel_error.abs() < self.epsilon as f64 {
                break;
            }
        }
        // println!("pi {:#?}", self.pi);
        // println!("gaussians {:#?}", self.gaussians);
    }

    fn predict(&mut self, test_samples: &Vec<f64>) -> Vec<usize> {

        let test_gamma_norm = test_samples
            .par_iter()
            .map(|&x| self.gaussians
                .iter()
                .zip(self.pi.iter())
                .map(|(gaussian, &p)| p * gaussian.pdf(x))
                .sum()
            ).collect::<Vec<f64>>();
            
        let test_gamma = test_samples
            .par_iter()
            .zip(test_gamma_norm.par_iter())
            .map(|(&x, cst)| self.gaussians
                .iter()
                .zip(self.pi.iter())
                .map(|(gaussian, &p)| p * gaussian.pdf(x)/cst)
                .collect::<Vec<f64>>()
            ).collect::<Vec<_>>();

        let test_z = test_gamma
            .par_iter()
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
pub struct MVN {
    dim: usize,
    mean: Vector3<f64>,
    cov: Matrix3<f64>,
    det_sqrt: f64,
    inv: Matrix3<f64>,
    cst: f64,
}


pub fn init_mvn( dim: usize, mean: Vector3<f64>, cov: Matrix3<f64>) -> MVN {
        
        let chol = cov.clone().cholesky().unwrap();
        let det_sqrt = chol.l().determinant();
        let cst = 1./((2.* std::f64::consts::PI).powf(dim as f64 / 2.) * chol.l().determinant());

        return MVN {
            dim: dim,
            mean: mean,
            cov: cov,
            det_sqrt: det_sqrt,
            inv: chol.inverse(),
            cst: cst,
        }
    }

impl MVN {
    pub fn pdf(&self, x: &Vector3<f64>) -> f64 { 
        return self.cst*(-0.5*(x-self.mean).transpose()*self.inv*(x-self.mean))[0].exp()
     }
}



pub struct GMM3D<'a> {
    nb_samples: usize,
    nb_components: usize,
    nb_iter: usize,
    epsilon: f64,
    samples: &'a Vec<Vector3<f64>>,
    pub pi: Vec<f64>,
    pub means: Vec<Vector3<f64>>,
    pub covs: Vec<Matrix3<f64>>,
    pub cov_reg: Matrix3<f64>,
    pub mvns: Vec<MVN>,
    pub z: Vec<usize>,
    weights: Vec<f64>,
    gamma_norm: Vec<f64>,
    gamma: Vec<Vec<f64>>,
    pub log_likelihoods: Vec<f64>,
}


pub fn init_gmm3_d<'a>( nb_samples: usize,
    nb_components: usize,
    cov_reg: f64,
    nb_iter:usize,
    epsilon:f64,
    samples: &'a Vec<Vector3<f64>>) -> GMM3D<'a>{
        return GMM3D {
            nb_samples: nb_samples,
            nb_components: nb_components,
            nb_iter: nb_iter,
            epsilon: epsilon,
            samples: &samples,
            pi: Vec::<f64>::new(),
            means: Vec::<Vector3<f64>>::new(),
            covs: Vec::<Matrix3<f64>>::new(),
            cov_reg: Matrix3::from_diagonal_element(cov_reg),
            mvns: Vec::<MVN>::new(),
            z: Vec::<usize>::new(),
            weights: Vec::<f64>::new(),
            gamma_norm: Vec::<f64>::new(),
            gamma: Vec::<Vec::<f64>>::new(),
            log_likelihoods: Vec::<f64>::new(),
        }
    }


impl<'a> EM<Vector3<f64>> for GMM3D<'a> {

    fn initialization(&mut self) {

        let mut rng = rand::thread_rng();
        // let uniform_covs = Uniform::new(0.05_f64, 0.5_f64).unwrap();
        let uniform_means = Uniform::new(-1_f64, 1_f64).unwrap();

        self.weights = vec![0 as f64; self.nb_components];
        self.gamma_norm = vec![0 as f64; self.nb_samples];
        self.gamma = vec![vec![0 as f64; self.nb_components]; self.nb_samples];
        self.z = vec![0 as usize; self.nb_samples];


        self.pi = (0..self.nb_components)
            .map(|_| 1 as f64 / self.nb_components as f64)
            .collect::<Vec<_>>();
        
        self.means = (0..self.nb_components)
            .map(|_| Vector3::from_vec(uniform_means.sample_iter(&mut rng)
                .take(3)
                .collect::<Vec<_>>()))
            .collect::<Vec<_>>();

        self.covs = (0..self.nb_components)
            .map(|_| Matrix3::from_diagonal_element(0.05))
        .collect::<Vec<_>>();

        self.mvns = self.means
            .iter()
            .zip(self.covs.iter())
            .map(|(&mean, &cov)| init_mvn(3, mean.clone(), cov.clone()))
            .collect::<Vec<_>>();

        // println!("pi {:#?}", self.pi);
        // println!("means {:#?}", self.means);
        // println!("covs {:#?}", self.covs);

    }

    fn normalization(&mut self) {
        // for i in 0..self.nb_samples {
        //     self.gamma_norm[i] = 0 as f64;
        //     for j in 0..self.nb_components {
        //         let mvn = MultivariateNormal::from_mean_and_covariance(&self.means[j], &self.covs[j]).unwrap();
        //         self.gamma_norm[i] += self.pi[j] * mvn.pdf(&(self.samples[i]).transpose())[0];
        //     }
        // }
        
        
        self.gamma_norm = self.samples
            .par_iter()
            .map(|&x| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)| p * mvn.pdf(&x))
                .sum()
            ).collect::<Vec<f64>>();
    }

    fn expectation(&mut self) {
        
        // for i in 0..self.nb_samples {
        //     for j in 0..self.nb_components {
        //         let mvn = MultivariateNormal::from_mean_and_covariance(&self.means[j], &self.covs[j]).unwrap();
        //         self.gamma[i][j] = self.pi[j]*mvn.pdf(&(self.samples[i]).transpose())[0]/self.gamma_norm[i];
        //     }
        // }

        self.gamma = self.samples
            .par_iter()
            .zip(self.gamma_norm.par_iter())
            .map(|(&x, cst)| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)| p * mvn.pdf(&x)/cst)
                .collect::<Vec<f64>>()
            ).collect::<Vec<Vec<f64>>>();
    }

    fn maximization(&mut self) {
        
        // for i in 0..self.nb_samples {
        //     let mut max_value = std::f64::NEG_INFINITY;
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
            .par_iter()
            .map(|gamma| gamma
                .iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap()
            )
            .collect::<Vec<usize>>();

        for j in 0..self.nb_components {
            self.weights[j] = 0 as f64;
            for i in 0..self.nb_samples {
                self.weights[j] += self.gamma[i][j];
            }
        }

        self.pi = self.weights
            .iter()
            .map(|x| x / self.nb_samples as f64)
            .collect::<Vec<f64>>();

        for j in 0..self.nb_components {
            // self.pi[j] = self.weights[j]/self.nb_samples as f64;

            self.means[j] = Vector3::from_vec(vec![0.0;3]);
            for i in 0..self.nb_samples {
                self.means[j] += self.gamma[i][j]*self.samples[i];
            }
            self.means[j] /= self.weights[j];
    
    
            self.covs[j] = Matrix3::from_vec(vec![0.0;9]);
            for i in 0..self.nb_samples {
                self.covs[j] += self.gamma[i][j]*(self.samples[i]-self.means[j])*(self.samples[i]-self.means[j]).transpose();
            }
            self.covs[j] /= self.weights[j];
            self.covs[j] += self.cov_reg;

            self.mvns[j] = init_mvn(3, self.means[j].clone(), self.covs[j].clone());
        }


        // self.gaussians = self.means.par_iter()
        //     .zip(self.covs.par_iter())
        //     .map(|(&mean, &cov)| Normal::new(mean, cov).unwrap())
        //     .collect::<Vec<_>>();
    }

    fn log_likelihood(&mut self) {

        // // let mut sum_log = 0 as f64;
        // // for i in 0..self.nb_samples {
        // //     sum_log += self.gamma_norm[i].ln();
        // // }

        let sum_log = self.gamma_norm
            .par_iter()
            .map(|gamma_norm| gamma_norm.ln())
            .sum::<f64>();

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
            
            if rel_error.abs() < self.epsilon as f64 {
                break;
            }
        }
        // println!("pi {:#?}", self.pi);
        // println!("means {:#?}", self.means);
        // println!("covs {:#?}", self.covs);
    }

    fn predict(&mut self, test_samples: &Vec<Vector3<f64>>) -> Vec<usize> {
        

        // let nb_samples_test = test_samples.len();

        // let mut test_gamma_norm = vec![0.0; nb_samples_test];
        // for i in 0..nb_samples_test {
        //     test_gamma_norm[i] = 0 as f64;
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
            .par_iter()
            .map(|&x| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)|
                    p * mvn.pdf(&x)
                )
                .sum()
            ).collect::<Vec<f64>>();
            
        let test_gamma = test_samples
            .par_iter()
            .zip(test_gamma_norm.par_iter())
            .map(|(&x, cst)| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)| p * mvn.pdf(&x)/cst)
                .collect::<Vec<f64>>()
            ).collect::<Vec<Vec<f64>>>();

        let test_z = test_gamma
            .par_iter()
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