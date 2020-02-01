## From NormalVAE to BinaryVAE with intermediate steps
## for Emiel 


### Short note on the implemented models

* NVAE: vanilla VAE with Gaussian Normal distribution

	    python experiment.py --latent_dist normal
	
* BNVAE: vanilla VAE, gets exactly same results but is implemented with 
explicit Bernoulli samples this is more like a sanity check

	    python experiment.py --latent_dist bnormal

* BVAE: BinConcrete VAE as implemented in \[Maddison, 2017\]

	    python experiment.py --latent_dist binary

* BCVAE: BinConcrete latent distribution, but samples are turned into 
continious samples

	    python experiment.py --latent_dist bcontinuous


### Requirements

This code has been tested with
-   `python 3.6`
-   `tensorflow 2.1.0` 
-   `tensorflow-probability 0.9.0` 
-   `matplotlib 3.1.2` 


Install conda environment via


	conda env create -f environment.yml 
	source activate binary_vae


### Approximate Bernoulli implementation
 
Tensorflow implmentation of  Bininary Concrete (BinConcrete) latent distribution, based on:

["The concrete distribution: A continuous relaxation of discrete random variables"](https://arxiv.org/pdf/1611.00712.pdf)
Maddison, Chris J., Andriy Mnih, and Yee Whye Teh, ICLR, 2017
