---
title: 'Where Did All the Memory Go?'
date: 2024-10-19
draft: true
comments: false
---

```CUDA error: out of memory```

This is a error not unfamilaiar to anyone who has tried to train a deep learnign model has faced, the usual solution is to just decrease the batch size and just move on, dont think about it too much....

In this blogpost i want to demystify, where all the memory is being consumed and some tricks that the industry has adopted, that you should be aware of to reduce the memory demands for training these memory hungry models.

Before we start discussing about the memory consumed buy the optimizer states, lest make sure everyone is on the same page, and get a quick refresher on the adam optimizer.

## Adam optimizer 
(if you are familiar with the adam optimizer you can skip this section)
### **1. Gradient Descent**

Before diving into Adam, let's briefly look at the foundational optimizer:

- **Gradient Descent (GD):**
- **Objective:** Minimize a loss function by iteratively updating parameters in the direction of the steepest descent (negative gradient).
- **Update Rule:**
$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t)
$$
where:
- $\theta_t$ are the parameters at iteration $t$
- $\eta$ is the learning rate
- $\nabla L(\theta_t)$ is the gradient of the loss function with respect to $\theta_t$

**Limitations**
- **Fixed Learning Rate:** Choosing an appropriate learning rate can be challenging; too high may cause divergence, too low may slow down convergence.
- **No Adaptation:** Doesn't adapt the learning rate based on the geometry of the loss surface, potentially leading to inefficient updates.

### **2. Introducing Adam Optimizer**

**Adam (Adaptive Moment Estimation)** combines the advantages of two other extensions of GD: **Momentum** and **RMSProp**. It computes adaptive learning rates for each parameter by maintaining estimates of both the first moment (mean) and the second moment (uncentered variance) of the gradients.

#### **Key Concepts:**

1. **Momentum:**
- Helps accelerate GD in the relevant direction and dampens oscillations.
- Maintains an exponentially decaying average of past gradients.

2. **Adaptive Learning Rates:**
- Adjusts the learning rate for each parameter individually based on the historical gradients.
- Parameters with higher gradients receive smaller updates, and vice versa.

**Adam** effectively combines these by maintaining both moving averages and adapting learning rates accordingly.

#### **Adam's Update Mechanism:**

Adam maintains two estimates for each parameter $\theta$:

- **First Moment ($m_t$):** Estimate of the mean of the gradients.
- **Second Moment ($v_t$):** Estimate of the uncentered variance (mean of the squared gradients).

The update steps are as follows:

1. **Initialize Parameters:**
- Initialize $m_0 = 0$ (first moment)
- Initialize $v_0 = 0$ (second moment)
- Choose hyperparameters:
- Learning rate ($\eta$)
- Decay rates for the moment estimates ($\beta_1$ for $m_t$, $\beta_2$ for $v_t$)
- Small constant ($\epsilon$) to prevent division by zero

2. **At each iteration $t$:**

a. **Compute Gradient:**
$$
g_t = \nabla L(\theta_{t-1})
$$
where $L$ is the loss function.

b. **Update First Moment ($m_t$):**
$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$
- $m_t$ accumulates the gradients.

c. **Update Second Moment ($v_t$):**
$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$
- $v_t$ accumulates the squared gradients.

d. **Bias Correction:**
Since $m_t$ and $v_t$ are initialized at zero, they are biased towards zero, especially during the initial steps. To correct this bias:
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$
$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$
- $\hat{m}_t$ and $\hat{v}_t$ are the bias-corrected estimates.

e. **Update Parameters:**
$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$
- This step uses the adaptive learning rate for each parameter.

#### **Algorithm Summary:**

<div style="text-align: center;">
  <img src="/images/where/where_adam.png" alt="Comparison of 32-bit, 16-bit, and bfloat16 floating-point formats." style="display: block; margin: 0 auto; width: 50%;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Comparison of 32-bit, 16-bit, and bfloat16 floating-point formats.
  <a href="https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html" style="color: rgba(0, 0, 0, 0.6);">(Maxime Labonne)</a>
</p>
</div>

comming back to the discussion of memory...

so now we know what to store cause of the adam optimizer, the model weights, the gradients, and the optimizer states constitute the majority of the memory required for training.

but this is not the whole picture there is additional memory called termed Residual memory by the ZeRO pape, 
this memory is made up of Activations, Temporary Buffers, and Memory Fragments.

Activations, are 
The outputs of each sub-component (self-attention, feed-forward network) within each layer are activations.
The final output of each layer, which becomes the input to the next layer, is also an activation.

Large models with extensive architectures generate substantial activations. For example, training a 1.5B parameter GPT-2 model with a sequence length of 1,000 tokens and a batch size of 32 can consume around 60 GB of memory solely for activations.

To mitigate high memory usage, activation checkpointing (or recomputation) can be employed. This technique saves memory by storing only a subset of activations and recomputing the others as needed during the backward pass. While it reduces memory consumption (e.g., from 60 GB to 8 GB in the aforementioned example), it introduces a computational overhead of about 33% due to the additional recomputation.

so why do we need to store the activations, these are essential to for calucating the gradients in backprop, 
**temp buffers**
Temporary buffers store intermediate results during operations like gradient all-reduce (used in distributed training) and gradient norm computation(used for gradient clipping). These buffers are essential for efficiently performing complex computations and communication between devices.
For large models, temporary buffers can require significant memory. For instance, a 1.5B parameter model might need around 6 GB of memory for a flattened fp32 buffer used during gradient all-reduce operations.

**Memory Fragmentation**
Even after taking all of the above memory consumption into consideration, you migh still encounter the dreadful "CUDA error: out of memory". This happens, because the gpu allocates memory in blcoks, even if ther is space for some of your params, but not all in a block of memory you will have issues.
In scenarios involving very large models, memory fragmentation can lead to out-of-memory (OOM) errors despite having a considerable amount of available memory. For example, training extremely large models might fail with over 30% of memory still free, but unusable due to fragmentation.


We cant do much about the Residual memory consumption, it is just good to have an idea about where did the memory go.

So lets take a look back at the memory we know we can control, ie the model weights, gradients and optimizer states.

We can use FP32 to store all of them but, its not necessary there is a concept of mixed precision training, which is widely adoppted in the industry, this inovlves stroing some of the components in lower precison FP16 (ie, half precision). 

Lets take a look at this now.

<div style="text-align: center;">
  <img src="/images/multi_gpu/mp_training.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>


The state-of-the-art approach to train large models on the current generation of NVIDIA GPUs is via mixed precision (fp16/32) training, where parameters and activations are stored as fp16, enabling the use of the high throughput tensor core units on these GPUs.

During mixed-precision training, both the forward and backward propagation are performed using fp16 weights and activations. However, to effectively compute and apply the updates at the end of the backward propagation, the mixed-precision optimizer keeps an fp32 copy of the parameters as well as an fp32 copy of all the other optimizer states.

Mixed precision training of a model with $\Psi$ parameters using Adam requires enough memory to hold an $\text{fp16}$ copy of the parameters and the gradients, with memory requirements of $2\Psi$ and $2\Psi$ bytes respectively. In addition, it needs to hold the optimizer states: an $\text{fp32}$ copy of the parameters, momentum, and variance, with memory requirements of $4\Psi$, $4\Psi$, and $4\Psi$ bytes, respectively. Let’s use $K$ to denote the memory multiplier of the optimizer states, i.e., the additional memory required to store them is $K\Psi$ bytes. Mixed-precision Adam has $K = 12$. In total, this results in $2\Psi + 2\Psi + K\Psi = 16\Psi$ bytes of memory requirement. For a model such as GPT-2 with 1.5 Billion parameters, this leads to a memory requirement of at least 24 GB, which is significantly higher than the meager 3 GB of memory required to hold the $\text{fp16}$ parameters alone.

so the above is true from a paper from ICLR 2018, after that people tried to train models in full FP16 format, which can be fruitful for smaller models, but below is an excerpt from the BLOOM model training ppl, who wrote a blogpost on huggingface.

> ### BF16Optimizer
> Training huge LLM models in FP16 is a no-no.
> We have proved it to ourselves by spending several months training a 104B model which as you can tell from the tensorboard was but a complete failure. We learned a lot of things while fighting the ever diverging lm-loss:
>  <div style="text-align: center;">
>    <img src="/images/multi_gpu/tensorboard_hf_excerpt.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
>  <p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
>    Figure 1: Comparison of FP8 and BF16 formats. Source: 
>    <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
>  </p>
>  </div>
> and we also got the same advice from the Megatron-LM and DeepSpeed teams after they trained the 530B model. The recent release of OPT-175B too reported that they had a very difficult time training in FP16.


So as you might rememember form the qunatization blogpost about, different datatypes, and my parise of the BF16 datatype.
The above is true if you have a V100 or older gpus that do not support the BF16 data type, but knowing the above is important because of historical reason or you might want to rent out a cheaper gpu fro training.

So how is Mixed precision training done using the BF16 fromat, lets take a look.

<div style="text-align: center;">
  <img src="/images/multi_gpu/mp_bf16_training.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

Here's a more concise explanation of the mixed precision data flow using BFLOAT16:

1. Input: Activations from previous layer (L-1) in BFLOAT16.

2. Forward Pass:
   - Multiply BFLOAT16 activations with BFLOAT16 weights.
   - Accumulate results in FP32.
   - Quantize output to BFLOAT16 for next layer's input.

3. Backward Pass:
   a) Error Gradients:
      - Multiply BFLOAT16 error gradients with transposed weights.
      - Accumulate in FP32, then quantize to BFLOAT16.
   b) Weight Gradients:
      - Multiply error gradients with transposed activations.
      - Accumulate results in FP32.

4. Weight Update:
   - Add FP32 weight gradients to FP32 master weights.

5. Precision Management:
   - Store master weights in FP32.
   - Use BFLOAT16 for computations and storage where possible.
   - Perform critical accumulations in FP32.
   - Convert between FP32 and BFLOAT16 as needed (shown as 'Q' operations).

This approach balances computational efficiency (using BFLOAT16) with numerical stability (using FP32 for critical operations), enabling effective training of large neural networks.

So again this paper was published in 2019, and people now train the full model with the BF16 fromat since gpus like the A100 can support these kinds of formats.

Lets take a look at at an even smaller format FP8, now people do train llms on this format. (this is a more latest paper came out in 2022).
this paper uses fp8 for the paramaters and gradients, but generally the optimizer states need to be stored in BF16 or higher.