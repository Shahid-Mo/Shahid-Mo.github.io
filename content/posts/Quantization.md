---
title: 'Quantization in LLMS (Part 1): LLM.int8(), NF4'
date: 2024-09-11
draft: false
comments: true
---

# Introduction to Quantization

Whether you're an AI enthusiast looking to run large language models (LLMs) on your personal device, a startup aiming to serve state-of-the-art models efficiently, or a researcher fine-tuning models for specific tasks, **quantization** is a key technique to understand.

Quantization can be broadly categorized into two main approaches:

* **Quantization Aware Training (QAT):** This involves training the model with reduced precision, allowing it to adjust during the training process to perform well under quantized conditions.
* **Post Training Quantization (PTQ):** Applied after a model has already been trained, PTQ reduces model size and inference cost without needing to retrain, making it especially useful for deploying models efficiently.

In this post (and subsequent ones on this topic), we'll focus on PTQ. It's often a simpler starting point for quantization, providing a good balance between performance and implementation complexity. PTQ is particularly useful for deploying models to edge devices or serving them at lower hardware costs.


# Why Quantization?

The costs of training large language models (LLMs) are already high, but inference costs—running the model to generate responses—can far exceed training costs, especially when deploying at scale. For instance, inference costs for models like ChatGPT can surpass training costs within just a week. Quantization helps reduce these costs by enabling models to operate in lower precision, such as FP16 or even INT4, without significant performance loss.

Let's look at an example: suppose you want to run the LLaMA 3.1 8B model. What kind of memory and GPU would you need? If you load the model from Hugging Face, it will automatically be loaded in full precision (FP32). So, just to load the model weights, you would need:

$$
8 \times 10^9 \text{ parameters} \times 4 \text{ bytes per parameter} \div 1024^3 \text{ for GB } \approx 30 \text{ GB}
$$

This means you would need at least an A100 GPU with 40 GB of VRAM to load just the model weights. For fine-tuning, you would require around 90 GB, or a cluster of GPUs, and about 60 GB for inference.

This is where quantization can help. By loading the model in 8-bit, 4-bit, or even 2-bit precision, you can reduce your memory requirements by a factor of up to 16. If you load the same model in NF4 format (4 bits per parameter), you would need just around 4 GB for inference, and about 12 GB for fine-tuning, which can be done even on a free-tier T4 GPU on Google Colab.

# Understanding Precision in LLMs

If you're unfamiliar with terms like FP32 or 8-bit precision mentioned earlier, don’t worry—we’ll cover them in the next section.

<div style="text-align: center;">
  <img src="/images/quant_p1/fp32_fp16_bf16.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

## Starting with FP32
FP32 (32-bit floating point) is the most common datatype used to train deep learning models. If you don’t specify the datatype of tensors in PyTorch, FP32 is the default.

To understand how FP32 works and why it's often preferred, let’s consider a simple example: representing the value of $\pi$ (pi) to 10 decimal places.

### Precision in Deep Learning Models
Let's assume we want to represent $\pi$ with the first 10 decimal places:

$$
\pi \approx 3.1415926535
$$

While this example is simple, it illustrates the inner workings of different data types like BF16 and FP16 — and later, we’ll discuss FP8, supported by the latest NVIDIA H100 GPUs and similar hardware.

### How to Convert a Decimal Representation to FP32

To convert the decimal value of $\pi$ (3.1415926535) into its FP32 (single-precision floating-point) representation, follow these steps:

### 1. Understand the FP32 Structure

FP32 is composed of 32 bits, divided into three parts:

- **Sign bit (1 bit):** Indicates whether the number is positive or negative.
- **Exponent (8 bits):** Encodes the exponent using a biased format.
- **Mantissa (23 bits):** Represents the significant digits of the number.

### 2. Convert $\pi$ to Binary

We start by converting the decimal value of $\pi$ into its binary equivalent.

**Convert the Integer Part:**

The integer part of $\pi$ is 3. In binary:

$$
3_{10} = 11_{2}
$$

**Convert the Fractional Part:**

To convert the fractional part (0.1415926535) to binary:

- Multiply by 2 and record the integer part repeatedly:
  - $0.1415926535 \times 2 = 0.283185307 \rightarrow 0$
  - $0.283185307 \times 2 = 0.566370614 \rightarrow 0$
  - $0.566370614 \times 2 = 1.132741228 \rightarrow 1$
  - $0.132741228 \times 2 = 0.265482456 \rightarrow 0$
  - $0.265482456 \times 2 = 0.530964912 \rightarrow 0$
  - Continue this process to generate more bits.

The binary representation of $\pi$, up to a reasonable precision, is approximately:

$$
\pi \approx 11.001001000011111101101010100010_{2}
$$

### 3. Normalize the Binary Representation

To fit the FP32 format, normalize the binary number so that it appears as:

$$
11.001001000011111101101010100010_{2} = 1.1001001000011111101101010100010_{2} \times 2^1
$$

### 4. Determine the Sign Bit

Since $\pi$ is positive, the sign bit is:

$$
\text{Sign bit} = 0
$$

### 5. Calculate the Exponent


In FP32, the exponent is stored with a bias of 127. This bias allows FP32 to represent both very large and very small numbers. Using FP32, the largest representable number is approximately $3.4 \times 10^{38}$, while the smallest positive number (close to zero) is around $1.175 \times 10^{-38}$. For $\pi$, the exponent after normalization is:

$$
\text{Exponent} = 1 + 127 = 128
$$

In binary, the exponent is:

$$
128_{10} = 10000000_{2}
$$

### 6. Determine the Mantissa

The mantissa consists of the significant digits after the leading 1:

$$
\text{Mantissa} = 10010010000111111011010_{2}
$$

Only the first 23 bits are kept; the rest are truncated, resulting in a small loss of precision. FP32 typically provides about 7 to 8 decimal places of precision. If more precision is needed, FP64 (double-precision) could be used, but for deep learning, FP32 is often more than sufficient.

### 7. Combine the Components

Now, combine the sign bit, exponent, and mantissa to form the final FP32 representation:

- **Sign bit:** 0
- **Exponent:** 10000000
- **Mantissa:** 10010010000111111011010

Thus, the FP32 representation of $\pi$ is:

$$
\text{FP32} = 0\ 10000000\ 10010010000111111011010_{2}
$$

This is the IEEE 754 standard representation of $\pi$ in FP32 format.

## FP16
The FP16 format uses fewer exponent and mantissa bits compared to FP32, This reduction results in a smaller range of values that can be represented. Specifically, in **FP32**, the largest representable number is approximately **$3.4 \times 10^{38}$**, while the smallest positive normal number is **$1.175 \times 10^{-38}$**

In contrast, **FP16** reduces these limits: the largest representable number is approximately **$6.55 \times 10^{4}$**, while the smallest positive normal number is **$6.10352 \times 10^{-5}$**.

This reduction in range makes FP16 more limited when storing very large or very small values, which can affect the stability of models when when using FP16 for training.

**Implications for Model Training and Fine-Tuning**:

- **Loading FP32 Models in FP16**: Simply converting an FP32 model to FP16 to fit it onto a GPU can lead to issues. The reduced range means that large weights or activations from the FP32 model might exceed FP16’s maximum representable value, causing overflow and resulting in "NaN" (Not a Number) errors during training.

- **Practical Solutions**:
   - **Mixed Precision Training**: A common approach is to use FP16 for most computations while keeping certain critical variables, like weights or gradients, in FP32. This balances the memory and speed benefits of FP16 with the numerical stability of FP32.
   - **Gradient Clipping**: Implementing techniques like gradient clipping can help prevent gradients from becoming too large, mitigating the risk of overflow in FP16.
   - **Loss Scaling**: Adjusting the scale of loss values can help maintain precision during backpropagation, reducing the chances of underflow in FP16.


## BF16

BF16, also known as Brain Float 16, was developed by Google Brain and became popular around 2019. It’s now one of the most widely used formats for training and fine-tuning LLMs. The main advantage of BF16 is that it retains the same number of exponent bits as FP32, making it easier to load a model in 16-bit precision and proceed with fine-tuning without running into "NaN" errors.

While BF16 sacrifices some precision compared to FP32, [research](https://arxiv.org/abs/1905.12322) has shown that the difference in performance when training or fine-tuning models on BF16 vs FP32 is minimal across different domains. Given the tradeoff between slightly reduced precision and the substantial reduction in model size, using BF16 is often well worth it for modern deep learning tasks.

## More Data Types


## TF32 (TensorFloat 32)

<div style="text-align: center;">
  <img src="/images/quant_p1/FP_16_32_TF32.png" alt="FP16 vs TF32 comparison" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

TF32, or TensorFloat 32, is a special datatype introduced by Nvidia with the Ampere architecture (e.g., RTX 30 and A100 series) in 2020. Unlike FP16, which you can explicitly declare in PyTorch, TF32 is not directly accessible as a datatype in the framework. Instead, it is a precision format used specifically by the CUDA cores of the GPU for certain operations, particularly matrix multiplications.

From the Ampere architecture onwards, all calculations involving FP32 matrices are done as shown in the figure. When performing matrix multiplications, the GPU automatically converts FP32 matrices into TF32 precision. The actual multiplication is done using TF32, which retains FP32’s 8-bit exponent but reduces the mantissa to 10 bits. Once the operation is complete, the results are accumulated back into FP32 format for higher precision in the final outcome.

This process leverages TF32 for speed during calculations, while maintaining the precision of FP32 for the final result. 

<div style="text-align: center;">
  <img src="/images/quant_p1/TF32_explained.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

## FP8 (E4M3 and E5M2 Formats)

<div style="text-align: center;">
  <img src="/images/quant_p1/FP8.png" alt="FP8 vs BF16 comparison" style="display: block; margin: 0 auto;">
  <p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>


FP8 is an emerging precision format designed to accelerate deep learning training and inference beyond the 16-bit formats commonly used today. Introduced in recent research paper *[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)* from September 2022, FP8 comes with two encoding options: **E4M3** (4-bit exponent, 3-bit mantissa) and **E5M2** (5-bit exponent, 2-bit mantissa). FP8 was introduced as part of Nvidia's Hopper Architecture (H100 GPUs).

The main advantage of FP8 is its ability to drastically reduce memory and compute requirements while maintaining comparable performance to higher precision formats like FP16 and BF16. The flexibility between the E4M3 and E5M2 formats allows FP8 to allocate an additional bit either to range or precision, depending on the task requirements.

As shown in the figures:
- **FP8 calculations**: Input matrices are converted to FP8, multiplied, and then accumulated in a higher precision format (FP16 or FP32). This allows for efficient computation while retaining enough precision in the final result. 
- **Efficacy of FP8**: The graph below illustrates the training loss (perplexity) across various large models (GPT-3 variants with up to 175 billion parameters), showing that FP8 training closely matches the results achieved with BF16. 

<div style="text-align: center;">
  <img src="/images/quant_p1/fp8_vs_bf16.png" alt="FP8 vs BF16 comparison" style="display: block; margin: 0 auto;">
  <p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

#### FP8 in Practice

In the paper, FP8 was used for training a wide variety of models, including CNNs, RNNs, and Transformer-based architectures, covering both image and language tasks. No changes were made to the model architectures, optimizer settings, or hyperparameters. FP8 post-training quantization (PTQ) was also evaluated, and it showed that models could be quantized to FP8 without a significant loss in accuracy, a major benefit for deployment.

### Nvidia is Everywhere!

Let's address the $2.5 trillion elephant in the room. Up until now, I’ve briefly mentioned that the TF32 datatype was introduced with Nvidia’s Ampere architecture, and that FP8 support starts with the H100 GPUs. We've been happily exploring the ins and outs of these different data types, but here’s the catch: You might get super excited about BF16 after reading this post and rush to try it out on a free Colab instance, only to realize the T4 GPUs don’t support BF16.

Don’t worry—I’ve done the hard work of sifting through Nvidia’s whitepapers so you don’t have to. Here's the deal: you need to know exactly what hardware you’re working with and which data types it supports. This way, you can make an informed decision about which GPU cluster to dedicate your time to and, in the process, help Nvidia grow even richer (they’re truly the ones selling shovels in this AI gold rush). And no, unfortunately, you can’t go running to AMD or Intel for your GPU needs either.

Take a look at the chart below, and I’ll follow up with some key observations. The numbers represent FLOPs (or TOPs for INT data types), and the X’s indicate that the datatype isn’t supported by a particular architecture.

<style>
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 25px 0;
    font-size: 18px;
    font-family: 'Arial', sans-serif;
    text-align: center;
  }
  th, td {
    padding: 12px;
    border: 1px solid #ddd;
  }
  th {
    background-color: #4CAF50;
    color: white;
  }
  tr:nth-child(even) {
    background-color: #f2f2f2;
  }
  tr:hover {
    background-color: #ddd;
  }
  .not-supported {
    color: #FF6347;
    font-weight: bold;
  }
  .sparse-matrix {
    color: #4682B4; /* Different color to highlight sparse matrix values */
    font-weight: bold;
  }
</style>

<table>
  <thead>
    <tr>
      <th>GPU</th>
      <th>P100</th>
      <th>V100</th>
      <th>T4</th>
      <th>A100</th>
      <th>H100</th>
      <th>L40</th>
      <th>B100</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>VRAM</b></td>
      <td>16GB</td>
      <td>32GB</td>
      <td>16GB</td>
      <td>40GB*</td>
      <td>80GB</td>
      <td>48GB</td>
      <td>192GB</td>
    </tr>
    <tr>
      <td><b>Architecture</b></td>
      <td>Pascal</td>
      <td>Volta</td>
      <td>Turing</td>
      <td>Ampere</td>
      <td>Hopper</td>
      <td>Ada Lovelace</td>
      <td>Blackwell</td>
    </tr>
    <tr>
      <td><b>Release Date</b></td>
      <td>Apr-16</td>
      <td>May-17</td>
      <td>Sep-18</td>
      <td>May-20</td>
      <td>Mar-22</td>
      <td>Sep-22</td>
      <td>Mar-24</td>
    </tr>
    <tr>
      <td><b>FP32</b></td>
      <td>10.6</td>
      <td>8.2</td>
      <td>8.1</td>
      <td>19.5</td>
      <td>67</td>
      <td>90.5</td>
      <td>60</td>
    </tr>
    <tr>
      <td><b>TF32</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="sparse-matrix">312</td>
      <td class="sparse-matrix">989</td>
      <td class="sparse-matrix">181</td>
      <td class="sparse-matrix">1800</td>
    </tr>
    <tr>
      <td><b>FP16</b></td>
      <td></td>
      <td>130</td>
      <td>65</td>
      <td class="sparse-matrix">624</td>
      <td class="sparse-matrix">1979</td>
      <td class="sparse-matrix">362</td>
      <td class="sparse-matrix">3500</td>
    </tr>
    <tr>
      <td><b>BF16</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="sparse-matrix">624</td>
      <td class="sparse-matrix">1979</td>
      <td class="sparse-matrix">362</td>
      <td class="sparse-matrix">3500</td>
    </tr>
    <tr>
      <td><b>INT8</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td>130</td>
      <td class="sparse-matrix">1248</td>
      <td class="sparse-matrix">3958</td>
      <td class="sparse-matrix">724</td>
      <td class="sparse-matrix">7000</td>
    </tr>
    <tr>
      <td><b>INT4</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td>260</td>
      <td class="sparse-matrix">2496</td>
      <td></td>
      <td class="sparse-matrix">1448</td>
      <td></td>
    </tr>
    <tr>
      <td><b>FP8</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="sparse-matrix">3958</td>
      <td class="sparse-matrix">724</td>
      <td class="sparse-matrix">7000</td>
    </tr>
    <tr>
      <td><b>FP4</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="sparse-matrix">14000</td>
    </tr>
  </tbody>
</table>

### Some Observations

* All the values in blue are for sparse matrices.
* As you can see, TF32 preforamce compared to the FP32 is 30 times.
* INT4 and INT8 data types were introduce in the Turing architecture for inference.


# Integer Qunatization

As you have already seen the T4 and Subsequnet gpu archtecture suppor INT8 and INT4 datatypes, thes datatypes can be used for inferecne and can be blazingly fast, (Just to keep another thing on the back of you mind is that if you load you matrix in a particula data type like INT8, this dosent necessarily mena the compuatations are Happening in IN8, will expand on this later)

## ABS Max Quantization

**Absmax quantization** is a method used to convert floating-point values (like FP16) into 8-bit integer values. Here's how it works in simpler terms:

1. **Scaling factor calculation**: 
   The input tensor (a matrix of floating-point numbers, $X_{f16}$) is scaled based on its largest absolute value. To do this, the method computes the **maximum absolute value** of all the elements in the tensor, also called the **infinity norm** ($\|X_{f16}\|\_\infty$). The scaling factor, $s_{xf16}$, is then calculated as:
   $$
   s_{xf16} = \frac{127}{\|X_{f16}\|_\infty}
   $$
   The value 127 is used because the 8-bit integer range is between $-127$ and $127$.

2. **Quantization**:
   Once the scaling factor is computed, the input tensor values are multiplied by $s_{xf16}$ to map them into the 8-bit range:
   $$
   X_{i8} = \text{round}\left(127 \cdot \frac{X_{f16}}{\|X_{f16}\|_\infty}\right)
   $$
   This means each value in the tensor is scaled down proportionally to fit into the [-127, 127] range, and then rounded to the nearest integer to make it an 8-bit value.

3. **Effect**: 
   This process allows a tensor originally in a high precision format (FP16) to be represented using 8-bit integers, reducing memory usage and potentially speeding up computations at the cost of some precision.

In summary, absmax quantization shrinks a floating-point tensor into the 8-bit range by dividing all elements by the largest absolute value in the tensor, then multiplying by 127 and rounding.




### LLM int8

Abs Max Quantization works, so shouldn't we just wrap up quantization and move on? Not quite. In November 2022, *[Detmers et al.](https://arxiv.org/abs/2208.07339)* introduced LLM.int8(), a new quantization technique, addressing an important issue: Large language models (LLMs) have started to show impressive emergent behaviors like reasoning, in-context learning, and even few-shot problem solving—abilities that were being compromised by naive quantization methods.

<div style="text-align: center;">
  <img src="/images/quant_p1/Emergence.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

The figure points out that once models exceed 2.7 billion parameters, naive 8-bit quantization significantly degrades performance. The drastic performance drop is due to the presence of **outliers**—weights that are crucial for enabling key behaviors like reasoning and in-context learning. These outliers are vital for the model's performance, so preserving them during quantization is essential to prevent significant degradation.
This happens because outlier weights—which play a crucial role in driving the emergent behaviors of large models—get lost in the quantization process.

LLM.int8() proposes a two-part strategy to address this issue:

1. **Vector-wise Quantization**: Instead of quantizing the entire tensor with a single scaling constant, LLM.int8() quantizes vectors individually.  The challenge with using a single scaling constant per tensor is that just one outlier can distort the entire quantization process, reducing precision for all other values. By applying multiple scaling constants—one for each vector—this method ensures that outliers don’t interfere with the rest of the matrix.

2. **Mixed-Precision Decomposition**: The second component of LLM.int8() involves quantizing only the weights that fall within a defined threshold, typically $[-6, 6]$. Outliers that exceed this range are left in higher precision formats like FP16 or FP32. By doing this, LLM.int8() ensures that these critical outlier weights—responsible for much of the model's performance—are preserved in higher precision, avoiding the loss of accuracy that comes with squeezing them into lower precision formats.

### Mixed-Precision Decomposition: Diving Deeper

To handle even larger models—those beyond 6.7 billion parameters—LLM.int8() introduces **mixed-precision decomposition**. This technique specifically tackles the emergence of extreme outliers in a fraction of the model's features. LLM.int8() performs matrix multiplication for these outlier dimensions in FP16, while the rest of the matrix operates in 8-bit. This ensures that outliers, which constitute only a small percentage of the total features but have a major impact on the model’s accuracy, are treated with the precision they require.

---

This revised version keeps the content approachable while adding clarity and flow. How does it feel now?










LLM int8 proposes a two part soultion to the above proble:

First, instead of quantizing an entire tensor with a single scaling constant, it uses vector-wise quantization, applying different scaling constants to each vector in the tensor. The issue with using a single scaling constant is that even one large outlier can reduce the precision of all other values in the tensor. By quantizing vectors individually, we prevent this precision loss.

First instead of qunatizing the whole tensor in one go, we qunatize it vector by vector this increase the number of qunaitzation constants to sotre, but it is totally worth it!!!!!!! 
The main challenge with quantization methods that use a single scaling constant per tensor is that
a single outlier can reduce the quantization precision of all other values. As such, it is desirable to
have multiple scaling constants per tensor
ie in ABSMAX qunaitzation absolute value was taken for the enire matrix or tensor, but taking the absolute value of individual vectors is better than than takinf the abs of the whole matrix, this technique is followed in other qunatization techniques like GPTQ as well.
The second idea of LLM.int8() is Qunatize weights within a certain threshold (this is huperparamater that you can set, default=6), ie if the weights are above or below this threshold $[-6,6]$ these weights are not qunatized, and stored in FP16/FP32/BF16 precision.
**The Problem with Single Scaling Constants**

- **Outliers**: In large tensors, especially in transformer models beyond a certain size (e.g., models larger than 6.7 billion parameters), there can be individual elements with exceptionally large values (outliers).
- **Reduced Precision**: When using a single scaling constant, these outliers force the scaling factor to be small to accommodate the largest values. This compresses the dynamic range for the rest of the values, leading to a loss of precision for the majority of the tensor elements.
- **Impact on Model Performance**: Reduced precision can degrade the performance of the model because small differences in weights or activations can significantly impact the model's outputs.

## the Mathematical details

### Vector-wise Quantization

In vector-wise quantization, the main idea is to apply **different scaling constants** for each vector in the matrix multiplication to reduce the impact of outliers on quantization precision. The key mathematical concepts behind vector-wise quantization are:

1. **Matrix Setup**:
   - Let $ X_{f16} \in \mathbb{R}^{b \times h} $ be the input matrix (e.g., hidden states), where $ b $ is the batch size and $ h $ is the hidden size.
   - Let $ W_{f16} \in \mathbb{R}^{h \times o} $ be the weight matrix, where $ o $ is the output size.
   
   The goal is to quantize these matrices from FP16 to Int8, applying different scaling constants to different rows or columns.

2. **Quantization**:
   - Quantization scales the values in each vector of $ X_{f16} $ and $ W_{f16} $ into the range $[-127, 127]$. The scaling constant for each row in $ X_{f16} $ is denoted $ c_{x_{f16}} $, and for each column in $ W_{f16} $, it is $ c_{w_{f16}} $.
   
   This can be represented as:
   $$
   Q(X_{f16}) = \frac{127}{\|X\_{f16}\|\_{\infty}} X\_{f16}
   $$
   where $ \|X_{f16}\|_{\infty} $ is the infinity norm (the maximum absolute value of the vector).

3. **Matrix Multiplication**:
   The quantized matrix multiplication is then performed in Int8 precision:
   $$
   C_{i32} = Q(X_{f16}) \cdot Q(W_{f16})
   $$
   Here, $ C_{i32} $ represents the result of the Int8 matrix multiplication.

4. **Denormalization**:
   To recover the correct FP16 result after quantization, the matrix result must be **denormalized** by the outer product of the scaling constants:
   $$
   C_{f16} \approx \frac{1}{c_{x_{f16}} \otimes c_{w_{f16}}} \cdot C_{i32}
   $$
   where $ \otimes $ is the **outer product** of the row-wise scaling constants $ c_{x_{f16}} $ and column-wise scaling constants $ c_{w_{f16}} $. This product ensures that the entire matrix is scaled back appropriately.

5. **Benefits**:
   - **Increased Precision**: By using multiple scaling constants, vector-wise quantization reduces the impact of outliers confined to specific rows or columns, improving the precision of the quantized representation.
   - **Efficiency**: This method maintains computational efficiency by applying row- and column-wise scaling, which is less computationally expensive than applying different scaling to every individual element.

### Mixed-Precision Decomposition

While vector-wise quantization works well for most situations, **outliers**—specific dimensions with significantly larger values—can still affect overall precision, especially in very large models (e.g., 6.7B parameters or more). Mixed-precision decomposition addresses this by applying different precisions for different parts of the matrix.

1. **Outliers**:
   - **Outliers** refer to specific dimensions in the matrix where the values are consistently much larger than others. These outliers might be critical for model performance and require higher precision for accurate representation.
   - For example, in a matrix $ A_{f16} \in \mathbb{R}^{s \times h} $, outliers might occur in some columns across almost all sequences but are limited to a small set of feature dimensions.

2. **Mixed-Precision Decomposition**:
   The matrix is decomposed into two parts:
   - **High-precision part (FP16)**: The outlier dimensions are processed with higher precision.
   - **Low-precision part (Int8)**: The rest of the dimensions are quantized into Int8 for efficiency.

   Mathematically:
   $$
   C_{f16} \approx \sum_{h \in O} X^h_{f16} \cdot W^h_{f16} + S \cdot \sum_{h \notin O} X_{i8}^h \cdot W_{i8}^h
   $$
   where:
   - $ O $ is the set of outlier dimensions.
   - The first term computes matrix multiplication for outlier features in FP16 precision.
   - The second term computes matrix multiplication for regular features in Int8 precision.
   - $ S $ is the scaling factor for dequantizing the Int8 results.

3. **Outlier Identification**:
   Outliers are identified based on a threshold $ \alpha $. If any value in a specific feature dimension exceeds $ \alpha $, that dimension is considered an outlier and handled with higher precision.

4. **Benefits**:
   - **High Precision for Critical Elements**: By isolating and handling outliers in FP16, the method maintains precision where it matters most, preserving model performance.
   - **Memory Efficiency**: Since outliers are typically sparse (only about 0.1% of all feature dimensions), the memory overhead is minimal, leading to significant memory savings compared to full FP16 computation.

### Summary

- **Vector-wise Quantization** improves precision by applying different scaling constants to rows and columns in the matrix, reducing the effect of outliers on the entire matrix.
- **Mixed-Precision Decomposition** further improves efficiency and precision by handling outliers with high precision (FP16) and the rest of the matrix with low precision (Int8), balancing memory usage and accuracy in large models.

These techniques allow large-scale models to be efficiently quantized while minimizing performance degradation caused by outliers.




