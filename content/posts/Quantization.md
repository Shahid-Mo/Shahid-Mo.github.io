---
title: "Quantization of Language Models"
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

