<h1 align="center">
  <br>
	Efficient Video Prediction via Sparsely Conditioned Flow Matching
  <br>
</h1>
  <p align="center">
    <a href="https://araachie.github.io">Aram Davtyan</a> •
    <a href="https://www.cvg.unibe.ch/people/sameni">Sepehr Sameni</a> •
    <a href="https://www.cvg.unibe.ch/people/favaro">Paolo Favaro</a>
  </p>
<h4 align="center">Official repository of the paper</h4>

<h4 align="center">ICCV 2023</h4>

<h4 align="center"><a href="https://araachie.github.io/river/">Project Website</a> • <a href="https://arxiv.org/abs/2211.14575">Arxiv</a>

#
> **Abstract:** *We introduce a novel generative model for
> video prediction based on latent flow matching, an efficient
> alternative to diffusion-based models. In contrast to prior work,
> we keep the high costs of modeling the past during training
> and inference at bay by conditioning only on a small random
> set of past frames at each integration step of the image
> generation process. Moreover, to enable the generation
> of high-resolution videos and to speed up the training, we
> work in the latent space of a pretrained VQGAN. Furthermore,
> we propose to approximate the initial condition of the
> flow ODE with the previous noisy frame. This allows to reduce
> the number of integration steps and hence, speed up
> the sampling at inference time. We call our model Random
> frame conditioned flow Integration for VidEo pRediction,
> or, in short, RIVER. We show that RIVER achieves superior
> or on par performance compared to prior work on common
> video prediction benchmarks, while requiring an order of
> magnitude fewer computational resources.*

## Citation

The paper is to appear in the Proceedings of the International Conference on Computer Vision in 2023. 
In the meantime we suggest using the arxiv preprint bibref.

A. Davtyan, S. Sameni, P. Favaro. Efficient Video Prediction via Sparsely Conditioned Flow Matching. Technical Report, 2023.
  
## Code
  
Coming soon...
