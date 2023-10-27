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
