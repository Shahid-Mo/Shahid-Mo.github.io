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


###  FP8
Recent research, such as the "FP8 FORMATS FOR DEEP LEARNING" paper from September 2022, even shows that it's possible to train large language models (LLMs) using FP8 with just 2 or 3 bits of mantissa (depending on the E4M3 or E5M2 format).

{{< figure src="/images/quant_p1/fp8_vs_bf16.png" alt="FP8 vs BF16 comparison" caption="Figure 1: Comparison of FP8 and BF16 formats. Source: [Smith et al. (2023)](https://arxiv.org/abs/xxxx.xxxxx)" class="center" >}}



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