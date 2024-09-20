---
title: 'Quantization in LLMS (Part 1): LLM.int8(), NF4'
date: 2024-09-11
draft: false
---

## Introduction to Quantization

Whether you're an AI enthusiast looking to run large language models (LLMs) locally on your personal device, a multi-billion-dollar startup aiming to serve a state-of-the-art model to customers, or someone wanting to fine-tune models like LLaMA or Flux, **quantization** is a technique you need to understand.

Quantization can broadly be categorized into two types:

- **Quantization Aware Training (QAT):** Integrates quantization into the training process, allowing the model to adapt to reduced precision during training.

- **Post Training Quantization (PTQ):** Applies quantization after a model is already trained, focusing on reducing its size and inference cost without retraining.

In this post (and subsequent ones on this topic), we'll focus on Post Training Quantization (PTQ) — a method used to make trained models smaller and more efficient, particularly useful when deploying models to edge devices or serving models at lower hardware costs.

## Why Quantization?

Consider this: the costs of training an LLM are already high, but inference costs — running the model to generate responses — can far exceed training costs, especially when deploying at scale. For example, inference costs for models like ChatGPT can surpass training costs within a week. Quantization reduces these costs by allowing models to operate in lower precision, such as FP16 or even FP8, without significant performance loss.

Let’s start with a real-world example using the GPT-2 model from Hugging Face. We will quantize it to reduce its memory footprint and discuss how to efficiently convert FP32 to lower-precision formats like FP16 and beyond.

We will begin with some "back-of-the-napkin" calculations. Suppose you want to run the LLaMA 3.1 70B model. What kind of memory and GPU are you looking at? If you load the model from Hugging Face, it will automatically be loaded in full precision (FP32). Just to load the model weights, you will require:
$$
70 \times 10^9 \text{ parameters} \times 32 \text{ bits per parameter} \div 8 \div 1024^3 \approx 280 \text{ GB}
$$


So, you would need at least an H100 GPU with 80 GB of VRAM to load the model, and you wouldn't even be able to perform inference or fine-tuning. You would likely need to get a cluster for that.

This is where quantization can help you. By loading the model in 8-bit, 4-bit, or even 2-bit precision, you can reduce your memory requirements by a factor of up to 16. If you load the same model in NF4 format (4 bits per parameter), you would need just around 35 GB, which can be accomplished even on the free-tier T4 Colab.

## Understanding Precision in LLMs

If you're not familiar with terms like FP32 or 8-bit precision from the previous section, we're going to cover that next.

### Starting with FP32
FP32 (32-bit floating point) is the most common datatype used to train deep learning models. If you don’t specify the datatypes of tensors in PyTorch, FP32 is the default datatype.

To understand how FP32 works and why it is often preferred, let’s look at a simple example: representing $\pi$ (pi) up to 10 decimal places.

### Precision in Deep Learning Models

Let's assume we want to represent $\pi$ with the first 10 decimal places:

$$
\pi \approx 3.1415926535
$$

While this example is quite basic, it helps illustrate the inner workings of different data types, such as BF16 and FP16 — and in future posts, FP8 (which is supported by the latest NVIDIA H100 GPUs and similar hardware).

# How to Convert a Decimal Representation to FP32

To convert the decimal value of $\pi$ (3.1415926535) into its FP32 (single-precision floating-point) representation, we need to follow these steps:

## 1. Understand the FP32 Structure

FP32 is composed of 32 bits, divided into three parts:

- **Sign bit (1 bit):** Indicates whether the number is positive or negative.
- **Exponent (8 bits):** Encodes the exponent using a biased format.
- **Mantissa (23 bits):** Represents the significant digits of the number.

## 2. Convert $\pi$ to Binary

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
  - Continue this process to get more bits.

The binary representation of $\pi$ up to a reasonable precision is approximately:

$$
\pi \approx 11.001001000011111101101010100010_{2}
$$

## 3. Normalize the Binary Representation

To fit the FP32 format, normalize the binary number so that it appears as:

$$
11.001001000011111101101010100010_{2} = 1.1001001000011111101101010100010_{2} \times 2^1
$$

## 4. Determine the Sign Bit

Since $\pi$ is positive, the sign bit is:

$$
\text{Sign bit} = 0
$$

## 5. Calculate the Exponent

The exponent is stored with a bias of 127 in FP32. This bias allows the representation of both very large and very small numbers. In FP32, the largest number you can represent is approximately $3.4 \times 10^{38}$, while the smallest positive number (close to zero) is around $1.4 \times 10^{-45}$.

For $\pi$, the exponent after normalization is 1:

$$
\text{Exponent} = 1 + 127 = 128
$$

The exponent in binary is:

$$
128_{10} = 10000000_{2}
$$

## 6. Determine the Mantissa

The mantissa consists of the significant digits after the leading 1:

$$
\text{Mantissa} = 10010010000111111011010_{2}
$$

Only the first 23 bits are retained, and any extra bits are truncated, resulting in some loss of precision. FP32 typically provides around 7 to 8 decimal places of precision. If you require more precision, you could use FP64 (double-precision), but for deep learning, FP32 is often more than sufficient.


## 7. Combine the Components

Combine the sign bit, exponent, and mantissa to form the final FP32 representation:

- **Sign bit:** 0
- **Exponent:** 10000000
- **Mantissa:** 10010010000111111011010

Thus, the FP32 representation of $\pi$ is:

$$
\text{FP32} = 0\ 10000000\ 10010010000111111011010_{2}
$$

This is the IEEE 754 standard representation of $\pi$ in FP32 format.

# Python Code to Demonstrate FP32 and Other Precisions

Let's use Python to compare different floating-point representations of $\pi$ and calculate their precision errors:

```python
import torch

# Pi value up to 10 digits
pi_val = 3.1415926535

# FP32 (single precision)
pi_fp32 = torch.tensor(pi_val, dtype=torch.float32)

# BF16 (Brain Floating Point 16, half precision)
pi_bf16 = torch.tensor(pi_val, dtype=torch.bfloat16)
```
```
og:   pi = 3.1415926535
FP32: Pi = 3.1415927410125732, Error % = 0.0000027856%
BF16: Pi = 3.140625,           Error % = 0.0308013675%
```

Now that we have an understanding, we can look at the different floating point and int representations present used in Deep Learingin

{{< figure src="/images/quant_p1/FP_16_32_TF32.png" alt="FP8 vs BF16 comparison" caption="Figure 1: Comparison of FP8 and BF16 formats. Source: [Smith et al. (2023)](https://arxiv.org/abs/xxxx.xxxxx)" class="center" >}}

### TF32
TF32 is a special datatype introduced by Nvidia from the Ampear Architecture, this is not present in pytorch ,you cannot simply declare a variable as TF32 in pytorch like you can do FP16, This is a datatype that is specifically uded by the cuda cores of the GPU, as you can see from the figure below, Nvidia GPU convert the FP32 Tensors, into TF32 Tensors and perform the matrix multiplications, and this is significantly faster than doing calcualtion in the FP32 format. 
Now that we have an understand of the FP32 representation, The TF32 or TensorFloat 32,Intoroduce by Nvidia (2020) tf32 is primarily supported by NVIDIA GPUs, starting with the Ampere architecture (e.g., RTX 30 and A100 series). These GPUs have dedicated hardware for handling tf32 operations, leading to substantial performance improvements in deep learning workloads.  

{{< figure src="/images/quant_p1/TF32_explained.png" alt="FP8 vs BF16 comparison" caption="Figure 1: Comparison of FP8 and BF16 formats. Source: [Smith et al. (2023)](https://arxiv.org/abs/xxxx.xxxxx)" class="center" >}}

###  FP16 
As you can see from the figure, the FP16 format has fewer Exponent and Mantisa Bits, the reduction of the number of Manissa bits has the Effect of reducing the max and Min values that can be stored using this formtat the Max and min values fall from (fp32 numbere) to (fp16 number), and min value form, ...   .This is an important thing to understan cause Naively loading the FP32 model in FP16 to Fit onto you gpu and then start finetuning is a bad idea, had to learn this the hard way,
Becaue of the difference between the exponent in FP32 and FP16, you might enounter "NaN" errors while training this is becaue the value is outside the precision of the format you are using, one naive soultion is if you are using FP16 is to load the model in full FP32 precision and you are good to go, a more principles approacht would be to figure out you your gradients are exploding or vaninsingh, This is generally not the case for Transformer base architectues, I faced this problem when I was trying to finetune a UNet diffusion model with and loaded the model in 16 Bit prcision, and the authors of the models had originally trained it on 32 Bit Precision.

### BF16
BF16 also called as Brain Float 16, was develope by google brain and gained populatiry in 2019, currently the most popular foramt to Train and Finetune LLMs right now, The big advantage of this format is that it has smae number of Exponet Bits comapred to the FP32 Format so it Makse loading the model in 16 Bits and then Finetunign it very easy, you dont run into "Nan" Loss errors as well when fine Tuning, The tradeoff is you loose precision but this paper has shown, that ther is not much difference beteen training and finetuning a LLM on 16 Bits, when compared to 32 bits, and this trade of in precision to smaller model size is totally worth it.

### FP8
{{< figure src="/images/quant_p1/fp8_vs_bf16.png" alt="FP8 vs BF16 comparison" caption="Figure 1: Comparison of FP8 and BF16 formats. Source: [Smith et al. (2023)](https://arxiv.org/abs/xxxx.xxxxx)" class="center" >}}


Recent research, such as the "FP8 FORMATS FOR DEEP LEARNING" paper from September 2022, even shows that it's possible to train large language models (LLMs) using FP8 with just 2 or 3 bits of mantissa (depending on the E4M3 or E5M2 format).

# Nvidia Exists !!!!!!!!!

Lets just Adress the 2.5 Trillion dollar elephat in the room, So till now, i have just hinted that the TF32 datatype was introduced in the Ampear architecture, and that the FP8 datatype is supported from the H100 gpu series. and we have been blissfully studying and enjoying different and different data type, But hers the thing, you read my post and get excited by the BF32 architecture so much that you want to try it out so you load the free version of the colab, and realize the T4 architecture dosent support the BF16 architecture, Fear not i have done the gruling work of sifting through Nvidias whitepapers, so that you dont have to. So heres the thing you need to know exactly what Hardware you have and what datatype these hardware suppoort so that you know whcich GPU cluste to surrender your soul out to, and make Nvidia richer (They are truly selling shovels in a Gold Rush, the more intresting thing is we cant buy our shovels from AMD or Intel Either). So i want you to Contemplate the below chart and i will follow with some observations, The numbers represent FLops (or TOPs for the INT Data types), The X represent the datatype is not supported by this architecture, 


some commetnts about the Nvidia GPUs




# some Headline to make it all go away
<div style="text-align: center;">
  <img src="/images/quant_p1/fp8_vs_bf16.png" alt="FP8 vs BF16 comparison" style="display: block; margin: 0 auto;">
  <p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/quant_p1/FP_16_32_TF32.png" alt="FP16 vs TF32 comparison" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/quant_p1/TF32_explained.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/quant_p1/llm_int8.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/quant_p1/Emergence.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/quant_p1/nf4.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>


Here’s the table formatted in Markdown for your blog, which you can directly use if your blog platform supports Markdown syntax:

### GPU Performance Table

| GPU            | P100  | V100  | T4    | A100     | H100  | L40   | B100  |
|----------------|-------|-------|-------|----------|-------|-------|-------|
| **VRAM**       | 16GB  | 32GB  | 16GB  | 40GB\*   | 80GB  | 48GB  | 192GB |
| **Architecture** | Pascal | Volta | Turing | Ampere   | Hopper | Ada Lovelace | Blackwell |
| **Release Date** | Apr-16 | May-17 | Sep-18 | May-20   | Mar-22 | Sep-22 | Mar-24 |
| **FP32**       | 10.6  | 8.2   | 8.1   | 19.5     | 67    | 90.5  | 60    |
| **TF32**       | X     | X     | X     | 312      | 989   | 181   | 1800  |
| **FP16**       |       | 130   | 65    | 624      | 1979  | 362   | 3500  |
| **BF16**       | X     | X     | X     | 624      | 1979  | 362   | 3500  |
| **INT8**       | X     | X     | 130   | 1248     | 3958  | 724   | 7000  |
| **INT4**       | X     | X     | 260   | 2496     |       | 1448  |       |
| **FP8**        | X     | X     | X     | X        | 3958  | 724   | 7000  |
| **FP4**        | X     | X     | X     | X        | X     | X     | 14000 |

### How to Use This Table in Your Blog

- **Headers** are defined with the `|` symbol.
- **Values** are centered by using colons `:` at the start and end of the hyphens `---` for each column.
- **Cells marked with 'X'** indicate the data type is not supported by that GPU.
  
#### Additional Formatting Tips:

- To make this table "dank" or visually appealing, consider using a custom stylesheet for your blog or platform. You can add background colors, borders, or hover effects using CSS.
- **Use a monospaced font** for the table or apply a dark theme to your blog to match the style.
  
Feel free to copy and paste this Markdown into your blog editor!

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
      <td>312</td>
      <td>989</td>
      <td>181</td>
      <td>1800</td>
    </tr>
    <tr>
      <td><b>FP16</b></td>
      <td class="not-supported">X</td>
      <td>130</td>
      <td>65</td>
      <td>624</td>
      <td>1979</td>
      <td>362</td>
      <td>3500</td>
    </tr>
    <tr>
      <td><b>BF16</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td>624</td>
      <td>1979</td>
      <td>362</td>
      <td>3500</td>
    </tr>
    <tr>
      <td><b>INT8</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td>130</td>
      <td>1248</td>
      <td>3958</td>
      <td>724</td>
      <td>7000</td>
    </tr>
    <tr>
      <td><b>INT4</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td>260</td>
      <td>2496</td>
      <td class="not-supported">X</td>
      <td>1448</td>
      <td class="not-supported">X</td>
    </tr>
    <tr>
      <td><b>FP8</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td>3958</td>
      <td>724</td>
      <td>7000</td>
    </tr>
    <tr>
      <td><b>FP4</b></td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td class="not-supported">X</td>
      <td>14000</td>
    </tr>
  </tbody>
</table>



## some reandom heading to differentiate the two

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
      <td></td>
      <td></td>
      <td></td>
      <td>312</td>
      <td>989</td>
      <td>181</td>
      <td>1800</td>
    </tr>
    <tr>
      <td><b>FP16</b></td>
      <td></td>
      <td>130</td>
      <td>65</td>
      <td>624</td>
      <td>1979</td>
      <td>362</td>
      <td>3500</td>
    </tr>
    <tr>
      <td><b>BF16</b></td>
      <td></td>
      <td></td>
      <td></td>
      <td>624</td>
      <td>1979</td>
      <td>362</td>
      <td>3500</td>
    </tr>
    <tr>
      <td><b>INT8</b></td>
      <td></td>
      <td></td>
      <td>130</td>
      <td>1248</td>
      <td>3958</td>
      <td>724</td>
      <td>7000</td>
    </tr>
    <tr>
      <td><b>INT4</b></td>
      <td></td>
      <td></td>
      <td>260</td>
      <td>2496</td>
      <td></td>
      <td>1448</td>
      <td></td>
    </tr>
    <tr>
      <td><b>FP8</b></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>3958</td>
      <td>724</td>
      <td>7000</td>
    </tr>
    <tr>
      <td><b>FP4</b></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>14000</td>
    </tr>
  </tbody>
</table>




# Integer Qunatization

As you have already seen the T4 and Subsequnet gpu archtecture suppor INT8 and INT4 datatypes, thes datatypes can be used for inferecne and can be blazingly fast, (Just to keep another thing on the back of you mind is that if you load you matrix in a particula data type like INT8, this dosent necessarily mena the compuatations are Happening in IN8, will expand on this later)

Lets start with some tradition Int8 quantization techniques just for the sake of completeness.

## ABS Max Quantization

### LLM int8

### NF4 qunatization  

**NF4 Quantization** (NormalFloat 4-bit quantization) is a specialized quantization technique designed to optimally compress data that follows a zero-centered normal distribution. It builds on **Quantile Quantization**, a method that assigns equal numbers of input values to each quantization bin, making it an information-theoretically optimal data type.

### Overview of NF4 Quantization

1. **Quantile Quantization**: 
   Quantile quantization aims to ensure each quantization bin has an equal number of input values. It is considered optimal since it minimizes the information loss across all bins by distributing the data uniformly. However, computing exact quantiles is computationally expensive, which necessitates using fast quantile approximation algorithms like **SRAM quantiles**. These approximations can result in large quantization errors for outliers, which are often critical.

2. **Fixed Distribution Quantization**:
   When input tensors come from a distribution that is fixed except for a quantization constant, the quantile estimation becomes computationally feasible. Pretrained neural network weights typically have a zero-centered normal distribution with some standard deviation \(\sigma\). To use NF4 quantization, all weights are scaled to a fixed distribution range, which, for simplicity, is set to \([-1, 1]\).

3. **Transformation to Fixed Range**:
   The weights are transformed to this range by normalizing them using their standard deviation \(\sigma\), effectively transforming all distributions to a standard normal form, \(N(0, 1)\). After normalization, the weights are quantized using a special 4-bit data type, NF4, that minimizes quantization error for normally distributed data.

### Steps for NF4 Quantization

To compute the optimal quantization scheme for normally distributed data, we perform the following steps:

1. **Estimate Quantiles**: 
   Calculate the quantiles for a standard normal distribution \(N(0, 1)\). For a 4-bit (i.e., \(k = 4\)) data type, we need to determine \(2^{k} + 1 = 17\) quantiles. The quantiles \(q_i\) are defined by:

   \[
   q_i = \frac{1}{2} \left( Q_X \left( \frac{i}{2^k + 1} \right) + Q_X \left( \frac{i + 1}{2^k + 1} \right) \right)
   \]

   where \(Q_X(\cdot)\) is the quantile function (inverse cumulative distribution function) of the standard normal distribution \(N(0, 1)\). This step gives us the quantile boundaries for the theoretical normal distribution.

2. **Normalize Values**:
   These quantiles are then normalized into the target range \([-1, 1]\). The input tensor's values are rescaled accordingly by dividing by their absolute maximum value, ensuring that the quantization bins match the transformed distribution of the input tensor.

3. **Quantize the Input Tensor**:
   Rescale the standard deviation \(\sigma\) of the input tensor to match that of the data type (which fits within the range \([-1, 1]\)). The quantized values can now be represented in the desired 4-bit format. The quantization bins derived from the estimated quantiles ensure that the representation is information-theoretically optimal for zero-centered normal data.

### Addressing Zero Representation

A common problem with symmetric quantization schemes is the lack of an exact zero representation, which is crucial for efficient padding and zero-valued elements. To solve this:

- NF4 creates an **asymmetric data type** by separately estimating the quantiles for the negative and positive parts of the distribution:
  - For the negative part, calculate quantiles for \(2^{k-1}\) bins.
  - For the positive part, calculate quantiles for \(2^{k-1} + 1\) bins.
- The quantiles are then unified by removing one of the duplicate zeros that appear in both sets.

This results in an asymmetric, information-theoretically optimal data type called **NormalFloat \(NFk\)**, which ensures zero-centered representation with minimal error.

### Double Quantization (DQ)

To further reduce memory usage, **Double Quantization** (DQ) is introduced, which quantizes the quantization constants themselves:

1. **First Quantization**:
   The initial quantization uses 32-bit constants and a block size of 64. This introduces a memory overhead of \(32/64 = 0.5\) bits per parameter.

2. **Second Quantization**:
   The 32-bit quantization constants \(c_{FP32}^2\) are treated as inputs for a second quantization. An 8-bit float representation is used with a block size of 256, yielding new quantized constants \(c_{FP8}^2\) and another set of quantization constants \(c_{FP32}^1\).

3. **Memory Reduction Calculation**:
   By employing 8-bit floats and symmetric quantization (centered around zero), the memory footprint per parameter is reduced from:

   \[
   \frac{32}{64} = 0.5 \text{ bits} \rightarrow \frac{8}{64} + \frac{32}{(64 \cdot 256)} = 0.127 \text{ bits}
   \]

   This represents a reduction of \(0.373\) bits per parameter.

### Summary

**NF4 Quantization**:
- Optimally quantizes zero-centered normal distributions using an information-theoretically optimal 4-bit format.
- Utilizes quantile estimation, normalization to \([-1, 1]\), and asymmetric binning to minimize quantization error.
- Incorporates **Double Quantization** to reduce memory usage for quantization constants, leveraging a two-tier quantization approach.

By aligning the quantization bins with the natural distribution of neural network weights, NF4 quantization achieves a highly efficient and compact representation, maintaining performance while reducing memory requirements.




Based on your images, here's a detailed blog post draft using the same structure and flow:

---

### **NF4 Quantization: A Specialized Quantization Technique for Deep Learning**

---

**NF4 Quantization** is an specialized quantization technique that is designed to be information-theoretically optimal for data that follows a zero-centered normal distribution. This is particularly useful for neural network weights, which often have such distributions. NF4 builds upon the concept of **Quantile Quantization (QQ)**.

It was introduced by Tim Detmers, in the QLoRa, and is one of the most common Datatype for PFET techniques like QLoRa and Adapters. 
NF4 quantization is a powerful technique used in cutting-edge PEFT methods, particularly in QLoRA. Introduced by Dettmers et al. in their QLoRA paper, it has proven highly effective for the efficient fine-tuning of large language models, offering significant performance improvements with reduced computational costs.

### **Understanding the Basis: Quantile Quantization**

- **Quantile Quantization** aims to assign an equal number of values from the input tensor to each quantization bin.
- **Why Quantile Quantization is Considered Optimal**: QQ minimizes information loss by distributing the data uniformly across all bins. This approach ensures that the quantization error is spread evenly across the range of data values.

However, computing exact quantiles is **computationally expensive** (It works by estimating the Quantiles of the input tensors throught the emperical CDF). Therefore, quantile approximation algorithms like SRAM quantiles are often used.  
- **The Downside of Quantile Approximation**: Approximation can result in large quantization errors, especially for outliers, which can be critical in certain applications.

### **Fixed Distribution Quantization: Addressing the Challenges**

To mitigate the side effects of quantile approximation, we use **Fixed Distribution Quantization**.

- **How Fixed Distribution Quantization Helps**: When input tensors come from a known or fixed distribution, quantile estimation becomes computationally feasible. It provides a constant for quantization that can be more accurately estimated.

### **Applying NF4 Quantization**

Typically, neural network (NN) weights have a zero-centered normal distribution $ N(0, \sigma) $ with some standard deviation $ \sigma $.

To utilize NF4 quantization:

1. **Normalization**: All weights are scaled to a fixed distribution range, typically set to \([-1, 1]\) for simplicity.
   - The weights are converted to this range by normalizing them using their standard deviation \( \sigma \).
   - This process effectively transforms all distributions to a standard normal form \( N(0, 1) \).

2. **Quantization with NF4**: After normalization, the weights are quantized using a special 4-bit datatype, NF4, that minimizes quantization errors for normally distributed data.

### **Steps for NF4 Quantization**

1. **Estimate Quantiles**:
   - For a 4-bit datatype, determine \( 2^k + 1 = 17 \) quantiles.
   - These quantiles, \( q_i \), are defined to distribute the data optimally.

#### **Quick Sidebar: Understanding Quantiles in Probability Distributions**

**Quantiles** are values that divide a dataset or probability distribution into intervals with equal probabilities.

- Common quantiles include:
  - **Quartiles**: 4 equal parts.
  - **Deciles**: 10 equal parts.

### **Why Use NF4 Quantization?**

NF4 quantization offers several advantages:

- **Information-Theoretically Optimal**: Especially effective for data with a zero-centered normal distribution.
- **Minimized Quantization Error**: Reduces errors that could impact model performance, particularly for neural network weights.
- **Computational Efficiency**: Provides a balance between maintaining model accuracy and computational feasibility.

### **Comparing NF4 with Other Quantization Methods**

- **Uniform Quantization**: Divides the data range into equal intervals without considering the data distribution, leading to potential information loss.
- **Fixed-Point and Linear Quantization**: These methods do not account for the underlying distribution, making them less optimal compared to NF4 for normally distributed data.

### **Applications of NF4 Quantization**

NF4 quantization is especially useful in scenarios where neural network weights are normally distributed, such as in deep learning models. Benefits include:

- **Memory Efficiency**: Reduces the memory footprint of models.
- **Faster Inference**: Optimizes performance on hardware with limited precision.
- **Maintained Accuracy**: Keeps the model’s performance close to that of higher-precision representations.

### **Conclusion**

NF4 quantization is a powerful technique for optimizing neural networks with normally distributed weights. It offers an information-theoretically optimal way to quantize data while balancing computational cost and model accuracy. As deep learning continues to evolve, methods like NF4 quantization will play a critical role in deploying efficient and accurate AI models.

---

Feel free to adjust or add sections based on your audience's technical level or specific interests!



### Exact Qunantile Calculations

This section is optional (Read only if you want to know exactly how these Quantiles are calculated)

We will use, 2 bit qunatization for this explanation, the 4 bit quantization is exactly the same process, just some more Quantile calculations. First we will assuse we have out weights uniformly distributed instead of a normal distributions,(just to get out intuitions right.) 

#### Case 1: Uniform Distribution

So the assumptions here are that our data is evenly distributed between [-1,1], so we have 2 bits to represet them, so instead of naively representing them as [-1,-0.5], [-0.5,1],[1,0.5],[0.5,1] (this would be information theoriticall suboptimal), we can use the our 4 representation as 4 partitions, and qunatize our continious distribution this way, [-1,-0.6],[-0.6,-0.2],[-0.2,0.2],[0.2,0.6],[0.6,1],
where each bit represents the partition.

we can use the Formulas to arrive at the same qunatiles.

| Quantization Method | On the fly quantization | CPU | CUDA GPU | RoCm GPU (AMD) | Metal (Apple Silicon) | torch.compile() support | Number of bits | Supports fine-tuning (through PEFT) | Serializable with 🤗 transformers | 🤗 transformers support |
|---------------------|------------------------|-----|----------|----------------|-----------------------|------------------------|---------------|------------------------------------|-----------------------------------|-------------------------|
| AQLM                | <span style="color:red">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:red">●</span> | <span style="color:red">●</span> | <span style="color:green">●</span> | 1 / 2 | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> |
| AWQ                 | <span style="color:red">●</span> | <span style="color:red">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:red">●</span> | ? | 4 | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> |
| bitsandbytes        | <span style="color:red">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:red">●</span> | <span style="color:green">●</span> | 4 / 8 | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> |
| EETQ                | <span style="color:green">●</span> | <span style="color:red">●</span> | <span style="color:green">●</span> | <span style="color:red">●</span> | ? | 8 | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> |
| GGUF / GGML         | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> | 1 - 8 | See GGUF section | See GGUF section | See GGUF section |
| GPTQ                | <span style="color:red">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:red">●</span> | 2 - 3 - 4 - 8 | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> |
| HQQ                 | <span style="color:green">●</span> | <span style="color:red">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:red">●</span> | 1 - 8 | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> |
| Quanto              | <span style="color:green">●</span> | <span style="color:red">●</span> | <span style="color:green">●</span> | <span style="color:red">●</span> | <span style="color:red">●</span> | 2 / 4 / 8 | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> |
| FBGEMM_FP8          | <span style="color:green">●</span> | <span style="color:red">●</span> | <span style="color:green">●</span> | <span style="color:red">●</span> | <span style="color:green">●</span> | 8 | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> |
| torchao             | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:red">●</span> | <span style="color:red">●</span> | partial support (int4 weight only) | 4 / 8 | <span style="color:green">●</span> | <span style="color:green">●</span> | <span style="color:green">●</span> |


