---
transition: slide-up
layout: cover
---

## DDPM - Denoising Diffusion Probabilistic models
Diffusion models are a class of likelihood-based models that generate samples by gradually removing noise from a signal.

---
transition: slide-up
---

## DDPM | The diffusion process

* **Forward diffusion process**: Corrupt the image till it becomes Gaussian noise (<span color="yellow">Fixed process</span>)
* **Reverse diffusion process**: Gradually remove the noise till you recover the image (<span color="yellow">Learned process</span>)

<img src="/diagrams/level_1_dm.drawio.png" class="w-100% mt-10">

<!--
(1) There are T steps in total. (2) At the end of the forward process, we want the image to be complete Gaussian noise so that we are able to sample from it
-->

---
transition: slide-left
---

## DDPM | The diffusion process

* Both forward and reverse processes are formalized as a Markov chains, therefore:
    * $q(x_t \mid x_{t-1}, \dots, x_0) = q(x_t \mid x_{t-1})$
    * $p(x_{t-1} \mid x_t, \dots, x_T) = p(x_{t-1} \mid x_t)$

<img src="/diagrams/level_2_dm.drawio.png" class="w-100% mt-5">

---
layout: cover
transition: slide-up
---

## DDPM | Forward diffusion process
⚠️ Quite a bit of math involved.

---
layout: center
transition: slide-up
---

## DDPM | Notation, rules & hyperparameters

* $T$ = the number of steps to perform in both the forward and reverse diffusion process
* $x_0$ = the original image drawn from the dataset
* $x_1, \dots, x_T$ = the intermediate images with progressively more noise added
* $\beta_1, \dots, \beta_T$ = hyperparameters that control the amount of noise added at each step
* In general, $\beta_t \in [0,1]$ and $\beta_1 < \beta_2 < \dots < \beta_T$

---
transition: slide-up
---

## DDPM | Forward (Gaussian) diffusion process

* Mathematically convenient to use (isotropic) Gaussian transition probabilities

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_{t} \mid \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

* In practice, just use the reparameterization trick to perform a diffusion step:

$$
x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon \hspace{1cm} \epsilon \sim \mathcal{N}(O,I)
$$


<img src="\diagrams\reparameterization.drawio.png" class="w-90% mt-10">

---
transition: slide-left
---

## DDPM | A closed form solution

We can jump from the original image $x_0$ to an arbitrary diffusion step $x_t$ using the following closed form solution:

$$
q(x_t \mid x_0) = \mathcal{N}(x_t \mid \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t)I )
$$

Where $\bar\alpha_t = \alpha_1\alpha_2\dots\alpha_t$ and $\alpha_t = 1 - \beta_i$.

Using the reparameterization trick:

<img src="\diagrams\closed_form_1.drawio.png" class="w-90% mt-5">

---
layout: cover
transition: slide-up
---

## DDPM | Reverse diffusion process
🤖 Deep learning time!

---
transition: slide-up
---

## DDPM | Reverse transition probability

* We are interested in reversing the diffusion process, i.e. going from $x_t$ back to $x_{t-1}$
* An interesting result is:

$$
q\left(x_{t-1}\mid x_t, x_0\right) = \mathcal N( x_{t-1}; \tilde\mu(x_t, x_0), \tilde\beta_t I )
$$

$$
\tilde{\beta}_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t

\hspace{1cm}

\tilde\mu(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 +\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
$$

* We can reverse the forward process <span color='yellow'>if we know the starting image $x_0$</span>

<img src="\diagrams\closed_form_2.drawio.png" class="w-90%">

---
transition: slide-up
---

## DDPM | Reverse transition probability

<div class="flex flex-col justify-center h-90%">

* Problem: we don't have $x_0$ at inference time, we want to generate $x_0$!

<img src="\diagrams\closed_form_2_not_available.drawio.png" class="w-90% mt-5">

* But we can use this results in other ways...

</div>

---
transition: slide-left
---

## DDPM | Deep Neural Networks

<div class="h-90% flex flex-row justify-between items-center">

<div class="w-50%">

* We want to <span color='red'>remove the dependency from $x_0$</span> 

$$
q\left(x_{t-1}\mid x_t, x_0\right) = \mathcal N( x_{t-1}; \color{#dd6666}\tilde\mu(x_t, x_0)\color{white}, \tilde\beta_t I )
$$

* using a <span color='green'>Deep Neural Network</span>:

$$
p_\theta\left(x_{t-1}\mid x_t\right) = \mathcal N( x_{t-1}; \color{#44cb75}\mu_\theta(x_t, t)\color{white}, \tilde\beta_t I )
$$

* where $\color{#44cb75}\mu_\theta(x_t, t)\color{white} \approx \color{#dd6666}\tilde\mu(x_t, x_0)$

</div>

<div class="w-40%">

<img src="diagrams/unet.drawio.png"/>

</div>

</div>

---
layout: cover
transition: slide-up
---

## Training the network

🖥️ Prepare your GPUs.

---
transition: slide-up
---

## DDPM | Maximum likelihood

<div class="flex flex-col justify-center h-90%">

We want to maximize the likelihood of the model against a dataset $\{x^{(i)}\}_{i=1}^N$ of samples from our target distribution $x^{(i)} \sim q(x)$.

Since a diffusion model is a latent variable model, we can compute the likelihood by marginalization over the latent variables $x_1, \dots, x_T$:

$$
p_\theta(x_0) = \int p_\theta(x_0, x_{1:T}) \space dx_{1:T}
$$

**Problem**: The integral above is <span color='red'>intractable</span> due to the huge dimensionality of the latent variables.

</div>

---
transition: slide-up
---

## DDPM | Variational lower bound

<div class="flex flex-col justify-center h-90%">

* We can derive a variational lower bound to maximize instead (see <span color='yellow'>Appendix A</span> at the end): 

$$
\begin{split}
L_{\text{VLB}} &= 
-\underbrace{D_{KL}\left( q\left(x_T\mid x_0\right) \mid \mid p\left(x_T\right) \right)}_{L_T} -\\
&- \underbrace{\sum_{t=2}^T D_{KL}\left( q\left(x_{t-1}\mid x_t, x_0\right) \mid \mid p_\theta\left(x_{t-1} \mid x_t\right) \right)}_{L_{t-1}} + \underbrace{\log p_\theta\left(x_0 \mid x_1\right)}_{L_0}
\end{split}
$$

</div>

---
transition: slide-up
---

## DDPM | Variational lower bound

<div class="flex flex-col justify-center h-90%">

* The <span color='red'>highlighted term</span> measures the difference between the distribution of $x_T$ coming from our fixed forward diffusion process and the normal distribution $p(x_T) = \mathcal{N}(0,I)$. Since it is true by construction, this term is constant and <span color='red'>we can just remove it</span>.

$$
\begin{split}
L_{\text{VLB}} &= \color{#dd6666}
-\underbrace{D_{KL}\left( q\left(x_T\mid x_0\right) \mid \mid p\left(x_T\right) \right)}_{L_T} \color{white}-\\
&- \underbrace{\sum_{t=2}^T D_{KL}\left( q\left(x_{t-1}\mid x_t, x_0\right) \mid \mid p_\theta\left(x_{t-1} \mid x_t\right) \right)}_{L_{t-1}} + \underbrace{\log p_\theta\left(x_0 \mid x_1\right)}_{L_0}
\end{split}
$$

</div>

---
transition: slide-up
---

## DDPM | Variational lower bound

<div class="flex flex-col justify-center h-90%">

* Every term in this lower bound is computable:

$$
L_{\text{VLB}} = - \underbrace{\sum_{t=2}^T D_{KL}\left( \color{#44cb75}q\left(x_{t-1}\mid x_t, x_0\right)\color{white} \mid \mid \color{#22d3ee}p_\theta\left(x_{t-1} \mid x_t\right)\color{white} \right)}_{L_{t-1}} + \underbrace{\log \color{#22d3ee}p_\theta\left(x_0 \mid x_1\right)\color{white}}_{L_0}
$$

* The term in <span color='green'>green</span> has a closed form solution, while terms in <span color='cyan'>blue</span> are computed using the neural network.

</div>

---
transition: slide-left
---

## DDPM | Training loop

<div class="flex flex-col justify-center h-90%">

The training loop is the following:

* Pick an image $x_0$ from the dataset 
* Draw a diffusion step $t$ randomly from ${1, \dots, T}$
* Compute the loss term $-L_{t-1}$
* Update the weights $\theta$ of the U-Net by gradient descent 

</div>

---
layout: cover
transition: slide-up
---

## Sampling process

🖼️ Generating brand new images!

---
transition: slide-up
---

## DDPM | Sampling

<div class="flex justify-center">
<img src="diagrams/sampling.drawio.png" class="w-100 mt-10">
</div>


---
layout: cover
transition: slide-up
---

## Simplifying everything by predicting the noise

🪄 Seems like magic, but it's still math.


---
layout: cover
transition: slide-up
---

## Conditional generation

🏎️ Driving our generative model!


---
transition: slide-up
---

## Diffusion models | Conditional generation

* Objective: learn a conditional distribution $p(x \mid y)$, which enable us to explicitly control the data we generate through conditioning information y.

<div class="flex flex-row justify-around">

<div class="flex flex-col justify-center items-center">
<p class="remove-par-m w-70 text-center">

$y$ as `class label`

</p>
<img src="/hamburger.png" class="w-50"/>

<p class="remove-par-m text-center"> 

$y=\text{hamburger}$ 

</p>
</div> 

<div class="flex flex-col justify-center items-center">
<p class="remove-par-m w-70 text-center">

$y$ as `text-prompt`

</p>
<img src="/bear-prompt.png" class="w-40 text-center"/>

<p class="remove-par-m w-100 text-center"> 

$$
y= \text{``Teddy bears working} \\
\text{underwater with 1990s technology"} 
$$ 

</p>

</div> 

</div>

---
transition: slide-up
---

## Diffusion models | Conditional generation

<div class="flex flex-row mt-5">
    <div class="flex flex-col items-center" v-click>
        <h3>Classifier guidance</h3>
        <img src="/diagrams/classifier-guidance.drawio.png" class="mt-5 w-90%"/>
    </div>
    <div class="flex flex-col items-center" v-click>
        <h3>Classifier-free guidance</h3>
        <img src="/diagrams/classifier-free-guidance.drawio.png" class="mt-5 w-90%" />
    </div>
</div>

---
transition: slide-left
---

## Diffusion models | Comparison

<div class="flex flex-row flex-wrap justify-around">

<div class="m-2">

GAN
* <span class="text-green">High quality samples</span>
* <span class="text-green">Fast sampling process</span>
* <span class="text-red">Low diversity</span>

</div>

<div class="m-2">

VAE
* <span class="text-red">Low quality samples</span>
* <span class="text-green">Fast sampling process</span>
* <span class="text-green">High diversity</span>

</div>

<div class="m-2">

Diffusion models
* <span class="text-green">High quality samples</span>
* <span class="text-red">Slow sampling process</span>
* <span class="text-green">High diversity</span>

</div>

<div class="flex flex-col justify-center items-center w-100%">
    <img src="/diagrams/generative-trilemma.drawio.png" class="w-50">
    <small>Generative learning trilemma (Xiao et al., 2022)</small>
</div>

</div>
