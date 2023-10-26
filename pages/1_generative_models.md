---
transition: slide-up
---

## Deep generative models

Given a dataset of samples $x$ drawn from $p(x)$, learn a model $p_\theta(x) \approx p(x)$ such that we can:

* generate new samples $x \sim p_\theta(x)$
* estimate the probability density of the sample $p_\theta(x)$

<div class="flex justify-center">
    <img src="/generative-models.drawio.png" class="w-140"/>
</div>

---
transition: slide-left
---

## Real-life examples of generative modeling 

<div class="flex flex-row flex-wrap justify-around mt-2">

<div class="w-45% flex flex-col justify-center">
    <img src="/examples/midjourney.webp" class="h-38">
    <small class="example-title">Midjourney - Text to image</small>
</div>

<div class="w-45% flex flex-col justify-center">
    <img src="/examples/get3d.gif" class="h-38">
    <small class="example-title">Get3D - 3D shape generation</small>
</div>

<div class="w-45% flex flex-col justify-center mt-5">
    <img src="/examples/magvit.gif" class="h-38">
    <small class="example-title">MAGVIT - Video generation</small>
</div>

<div class="w-45% flex flex-col justify-center mt-5">
    <img src="/examples/spotify_voice_translation.gif" class="h-38">
    <small class="example-title">Spotify’s AI Voice Translation Pilot - Voice Translation</small>
</div>

</div>