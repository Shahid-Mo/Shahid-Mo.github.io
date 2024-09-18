+++
title = 'Quantization in LLMS Part 2: GPTQ'
date = 2024-08-01T09:51:17-04:00
draft = false
math = true
+++

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





**Introduction**

Quantization is a crucial technique in deep learning that reduces the memory footprint and computational requirements of neural networks by representing weights and activations with lower-precision numerical formats. This is particularly important when deploying large models on devices with limited resources. However, quantizing a neural network without significantly degrading its performance is challenging.

The GPTQ (Gradient Post-Training Quantization) algorithm is a method designed to efficiently quantize large-scale neural networks, such as those used in natural language processing, while maintaining high accuracy. GPTQ builds upon previous methods like Optimal Brain Quantization (OBQ) but introduces significant modifications to make it scalable to models with billions of parameters.

In this explanation, we will delve into the mathematical foundations of GPTQ, explain how it leverages the Hessian matrix and its inverse, discuss the role of the Cholesky decomposition, and provide a detailed walkthrough of the algorithm. We will also include examples to illustrate key concepts.

---

### **1. Problem Statement**

Given a pre-trained neural network, our goal is to quantize its weights so that the network's output remains as close as possible to that of the original network when processing a set of inputs. Specifically, for a linear (fully connected) layer, we aim to find a quantized weight matrix $ \mathbf{W}_c $ that minimizes the reconstruction error.

**Mathematically**, the objective is:

$$
\underset{\mathbf{W}_c}{\text{argmin}} \, \| \mathbf{W}\mathbf{X} - \mathbf{W}_c \mathbf{X} \|_2^2
$$

- $ \mathbf{W} $: Original full-precision weight matrix of the layer.
- $ \mathbf{W}_c $: Quantized weight matrix we want to find.
- $ \mathbf{X} $: Input matrix to the layer (a set of $ m $ input examples).
- $ \| \cdot \|_2 $: Frobenius norm, summing over all elements.

**Goal**: Find $ \mathbf{W}_c $ that minimizes the output difference caused by quantization.

---

### **2. Optimal Brain Quantization (OBQ)**

#### **2.1. Row-wise Independent Quantization**

OBQ simplifies the problem by treating each row $ \mathbf{w} $ of $ \mathbf{W} $ independently. This is reasonable because in a fully connected layer, each output neuron corresponds to one row of $ \mathbf{W} $, and the neurons operate independently given the inputs.

**Objective per row**:

$$
\underset{\mathbf{w}_c}{\text{argmin}} \, \| \mathbf{w}\mathbf{X} - \mathbf{w}_c \mathbf{X} \|_2^2
$$

#### **2.2. Quadratic Formulation**

The error for each row can be expressed as:

$$
E(\mathbf{w}_c) = \| \mathbf{w}\mathbf{X} - \mathbf{w}_c \mathbf{X} \|_2^2 = (\mathbf{w} - \mathbf{w}_c) \mathbf{H} (\mathbf{w} - \mathbf{w}_c)^\top
$$

Where:

- $ \mathbf{H} = \mathbf{X}\mathbf{X}^\top $: The Hessian matrix for this quadratic form.
- $ \mathbf{w} $: Original weights (vector).
- $ \mathbf{w}_c $: Quantized weights (vector).

The Hessian $ \mathbf{H} $ captures the second-order derivatives of the error with respect to $ \mathbf{w}_c $.

#### **2.3. Greedy Quantization**

OBQ quantizes one weight at a time:

1. **Select weight to quantize**: Choose the weight that, when quantized, results in the smallest increase in the error $ E $.

2. **Update remaining weights**: Adjust the unquantized weights to compensate for the error introduced by quantizing the selected weight.

#### **2.4. Mathematical Derivation**

**Step 1: Quantization Error for a Single Weight**

Let’s consider quantizing the $ q $-th weight $ w_q $ in $ \mathbf{w} $:

$$
\delta w_q = w_q - \text{quant}(w_q)
$$

The change in the error due to quantizing $ w_q $ is:

$$
\Delta E = (\delta w_q)^2 H_{qq}
$$

Where $ H_{qq} $ is the $ q $-th diagonal element of $ \mathbf{H} $.

**Step 2: Updating Remaining Weights**

To minimize the error, we adjust the remaining unquantized weights $ \mathbf{w}_F $:

$$
\delta \mathbf{w}_F = -\frac{\delta w_q}{H_{qq}} \mathbf{H}_{Fq}
$$

- $ \mathbf{H}_{Fq} $: The $ q $-th column (excluding $ q $-th row) of $ \mathbf{H} $.

This adjustment aims to compensate for the error introduced by quantizing $ w_q $.

**Step 3: Update Hessian Inverse**

After quantizing $ w_q $, we need to update the inverse Hessian $ \mathbf{H}_F^{-1} $ for the remaining weights:

$$
\mathbf{H}_F^{-1} \leftarrow \mathbf{H}_F^{-1} - \frac{\mathbf{H}_F^{-1} \mathbf{e}_q \mathbf{e}_q^\top \mathbf{H}_F^{-1}}{H_{qq}}
$$

- $ \mathbf{e}_q $: Standard basis vector with 1 at position $ q $ and zeros elsewhere.

**Note**: This update uses the Sherman-Morrison formula for rank-one updates of matrix inverses.

---

### **3. Limitations of OBQ**

While OBQ is effective for small to medium-sized models, it faces challenges with large models:

- **Computational Complexity**: The algorithm has cubic time complexity $ O(d_{\text{row}} \cdot d_{\text{col}}^3) $, where $ d_{\text{row}} $ and $ d_{\text{col}} $ are the dimensions of $ \mathbf{W} $.

- **Memory Requirements**: Storing and updating the Hessian inverse becomes impractical for layers with millions of parameters.

---

### **4. GPTQ Algorithm**

GPTQ introduces several key modifications to make the quantization process scalable:

#### **4.1. Arbitrary Quantization Order**

**Insight**: Quantizing weights in an arbitrary fixed order (e.g., left to right) performs almost as well as the greedy order, especially for large layers.

**Benefit**: This allows all rows to quantize weights in the same order, making $ \mathbf{H} $ and its inverse the same across rows.

#### **4.2. Shared Hessian Inverse**

Since all rows share the same quantization order, we can compute $ \mathbf{H}^{-1} $ once and use it for all rows.

- **Computational Saving**: Reduces complexity from $ O(d_{\text{row}} \cdot d_{\text{col}}^3) $ to $ O(\max\{d_{\text{row}} \cdot d_{\text{col}}^2, d_{\text{col}}^3\}) $.

#### **4.3. Lazy Batch Updates**

To improve computational efficiency:

- **Process Blocks**: Quantize weights in blocks of columns (e.g., 128 columns at a time).

- **Batch Updates**: Update $ \mathbf{W} $ and $ \mathbf{H}^{-1} $ after processing each block, rather than after every weight quantization.

**Benefit**: Increases the compute-to-memory-access ratio, better utilizing GPU capabilities.

#### **4.4. Cholesky Decomposition**

To address numerical instability:

- **Observation**: Only certain parts (rows/columns) of $ \mathbf{H}^{-1} $ are needed during quantization.

- **Solution**: Use the Cholesky decomposition of $ \mathbf{H}^{-1} $ to compute necessary components more stably.

- **Cholesky Decomposition**: For a symmetric positive-definite matrix $ \mathbf{A} $, the Cholesky decomposition finds a lower triangular matrix $ \mathbf{L} $ such that $ \mathbf{A} = \mathbf{L} \mathbf{L}^\top $.

**Benefit**: Enhances numerical stability and reduces computational errors, especially important for large models.

---

### **5. Mathematical Details**

#### **5.1. Hessian Matrix $ \mathbf{H} $**

**Definition**:

$$
\mathbf{H} = 2 \mathbf{X} \mathbf{X}^\top + \lambda \mathbf{I}
$$

- $ \mathbf{X} \in \mathbb{R}^{d_{\text{col}} \times m} $: Input matrix with $ m $ examples.
- $ \lambda $: Damping factor added to ensure numerical stability (e.g., $ \lambda = 0.01 \times \text{mean of diagonal elements} $).
- $ \mathbf{I} $: Identity matrix.

**Role**: Captures the curvature of the error function with respect to the weights.

**Note**: The factor of 2 arises from the derivative of the squared error.

#### **5.2. Inverse Hessian $ \mathbf{H}^{-1} $**

- **Purpose**: Required to compute the optimal adjustments to the unquantized weights when quantizing a weight.

- **Computation**: Direct inversion is computationally expensive for large matrices.

#### **5.3. Cholesky Decomposition**

**Concept**:

- For a symmetric positive-definite matrix $ \mathbf{A} $, the Cholesky decomposition finds a lower triangular matrix $ \mathbf{L} $ such that:

$$
\mathbf{A} = \mathbf{L} \mathbf{L}^\top
$$

**Significance**:

- **Numerical Stability**: More stable than direct inversion or eigenvalue decomposition for positive-definite matrices.

- **Efficient Computation**: Allows solving linear systems $ \mathbf{A}\mathbf{x} = \mathbf{b} $ efficiently by forward and backward substitution.

**Application in GPTQ**:

- Instead of computing $ \mathbf{H}^{-1} $ directly, compute the Cholesky decomposition $ \mathbf{L} $.

- Use $ \mathbf{L} $ to compute necessary components of $ \mathbf{H}^{-1} $ when needed.

---

### **6. GPTQ Algorithm Steps**

#### **6.1. Initialization**

- **Compute Hessian Matrix**:

$$
\mathbf{H} = 2 \mathbf{X} \mathbf{X}^\top + \lambda \mathbf{I}
$$

- **Compute Cholesky Decomposition**:

$$
\mathbf{H}^{-1} = (\mathbf{L} \mathbf{L}^\top )^{-1}
$$

However, we don't compute $ \mathbf{H}^{-1} $ explicitly. Instead, we use $ \mathbf{L} $ to solve systems involving $ \mathbf{H}^{-1} $.

#### **6.2. Quantization Loop**

For each block of columns (weights), perform the following:

**Step 1: Quantize Weights**

- For each column $ j $ in the block, quantize $ \mathbf{W}_{:, j} $:

$$
\mathbf{Q}_{:, j} = \text{quant}(\mathbf{W}_{:, j})
$$

- Compute quantization error:

$$
\delta \mathbf{W}_{:, j} = \mathbf{W}_{:, j} - \mathbf{Q}_{:, j}
$$

**Step 2: Update Remaining Weights**

- Adjust unquantized weights to compensate for the error introduced by quantizing the current weight(s).

- Use the Cholesky factors to compute the necessary adjustments efficiently.

**Step 3: Batch Updates**

- After processing the block, update the remaining weights and the relevant parts of $ \mathbf{H}^{-1} $.

**Algorithm Pseudocode**:

```plaintext
Initialize Q = zeros(d_row, d_col)
Compute H = 2 * X * X^T + lambda * I
Compute Cholesky decomposition of H^-1: H_inv = Cholesky(H^-1)

For i in range(0, d_col, block_size):
    For j in range(i, i + block_size):
        Q[:, j] = quantize(W[:, j])
        E[:, j - i] = (W[:, j] - Q[:, j]) / H_inv[j, j]
        W[:, j:(i + block_size)] -= E[:, j - i] * H_inv[j, j:(i + block_size)]
    W[:, (i + block_size):] -= E * H_inv[i:(i + block_size), (i + block_size):]
```

---

### **7. Example**

Let's illustrate the GPTQ algorithm with a simplified example.

**Assumptions**:

- $ \mathbf{W} $ is a $ 2 \times 2 $ weight matrix.
- $ \mathbf{X} $ is a $ 2 \times 3 $ input matrix (3 examples).
- We quantize to 1-bit weights (e.g., $-1$ or $+1$).

**Step 1: Initialize**

- **Weights**:

$$
\mathbf{W} = \begin{bmatrix} 0.8 & -0.5 \\ 0.3 & 0.7 \end{bmatrix}
$$

- **Inputs**:

$$
\mathbf{X} = \begin{bmatrix} 0.2 & -0.1 & 0.4 \\ 0.5 & 0.3 & -0.2 \end{bmatrix}
$$

- **Compute Hessian**:

$$
\mathbf{H} = 2 \mathbf{X} \mathbf{X}^\top + \lambda \mathbf{I}
$$

Compute $ \mathbf{X} \mathbf{X}^\top $:

$$
\mathbf{X} \mathbf{X}^\top = \begin{bmatrix} (0.2)^2 + (-0.1)^2 + (0.4)^2 & 0.2*0.5 + (-0.1)*0.3 + 0.4*(-0.2) \\ \text{Symmetric} & (0.5)^2 + (0.3)^2 + (-0.2)^2 \end{bmatrix}
$$

Compute $ \mathbf{H} $ (assuming $ \lambda = 0 $ for simplicity).

- **Cholesky Decomposition**:

Compute $ \mathbf{L} $ such that $ \mathbf{H} = \mathbf{L} \mathbf{L}^\top $.

**Step 2: Quantization**

- **Quantize $ \mathbf{W} $**:

$$
\mathbf{Q} = \begin{bmatrix} \text{quant}(0.8) & \text{quant}(-0.5) \\ \text{quant}(0.3) & \text{quant}(0.7) \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}
$$

- **Compute Quantization Error**:

$$
\delta \mathbf{W} = \mathbf{W} - \mathbf{Q} = \begin{bmatrix} 0.8 - 1 & -0.5 + 1 \\ 0.3 - 1 & 0.7 - 1 \end{bmatrix} = \begin{bmatrix} -0.2 & 0.5 \\ -0.7 & -0.3 \end{bmatrix}
$$

**Step 3: Update Remaining Weights**

- Use $ \delta \mathbf{W} $ and $ \mathbf{H}^{-1} $ (via Cholesky factors) to adjust the unquantized weights.

- For this small example, adjustments are minor.

---

### **8. Understanding the Hessian in Pre-trained Models**

#### **8.1. Source of the Hessian**

In the context of quantization, the Hessian matrix $ \mathbf{H} $ arises from the second-order Taylor expansion of the error function with respect to the weights.

- **Error Function**:

$$
E(\mathbf{w}_c) = \| \mathbf{w}\mathbf{X} - \mathbf{w}_c \mathbf{X} \|_2^2
$$

- **First Derivative**:

$$
\frac{\partial E}{\partial \mathbf{w}_c} = -2 (\mathbf{w}\mathbf{X} - \mathbf{w}_c \mathbf{X}) \mathbf{X}^\top
$$

- **Second Derivative (Hessian)**:

$$
\mathbf{H} = \frac{\partial^2 E}{\partial \mathbf{w}_c^2} = 2 \mathbf{X} \mathbf{X}^\top
$$

#### **8.2. Interpretation**

- $ \mathbf{H} $ captures how sensitive the error is to changes in $ \mathbf{w}_c $.

- In quantization, we are interested in how quantizing a weight affects the overall error, and $ \mathbf{H} $ provides this information.

#### **8.3. Hessian Computation in Practice**

- For large models, computing $ \mathbf{H} $ directly is impractical.

- **Approximation**: Use a subset of data ($ m $ examples) to compute $ \mathbf{X} $ and hence $ \mathbf{H} $.

- **Regularization**: Add damping ($ \lambda \mathbf{I} $) to $ \mathbf{H} $ to ensure it is positive-definite and invertible.

---

### **9. Cholesky Decomposition in Detail**

#### **9.1. Mathematical Background**

- **Definition**: For a symmetric positive-definite matrix $ \mathbf{A} $, there exists a unique lower triangular matrix $ \mathbf{L} $ with positive diagonal elements such that:

$$
\mathbf{A} = \mathbf{L} \mathbf{L}^\top
$$

#### **9.2. Computation Steps**

1. **Initialization**:

   - Let $ \mathbf{A} $ be $ n \times n $.
   - $ \mathbf{L} $ is initialized as a zero matrix.

2. **Algorithm**:

   For $ i = 1 $ to $ n $:

   - Compute:

     $$
     L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2}
     $$

   - For $ j = i+1 $ to $ n $:

     $$
     L_{ji} = \frac{1}{L_{ii}} \left( A_{ji} - \sum_{k=1}^{i-1} L_{jk} L_{ik} \right)
     $$

3. **Result**:

   - $ \mathbf{L} $ is lower triangular.
   - $ \mathbf{L} \mathbf{L}^\top = \mathbf{A} $.

#### **9.3. Significance in GPTQ**

- **Numerical Stability**: Cholesky decomposition is numerically stable for positive-definite matrices.

- **Efficient Solves**: Allows solving linear systems $ \mathbf{A}\mathbf{x} = \mathbf{b} $ by:

  1. Forward substitution to solve $ \mathbf{L}\mathbf{y} = \mathbf{b} $.
  2. Backward substitution to solve $ \mathbf{L}^\top \mathbf{x} = \mathbf{y} $.

- **Avoids Explicit Inversion**: Inverting $ \mathbf{H} $ directly can be numerically unstable and computationally expensive.

---

### **10. Conclusion**

GPTQ is a powerful algorithm for quantizing large neural networks efficiently while maintaining high accuracy. By leveraging insights about quantization order, batching updates, and utilizing Cholesky decomposition, GPTQ addresses the computational and numerical challenges posed by large-scale models.

**Key Takeaways**:

- **Hessian Matrix**: Central to understanding how quantization errors propagate and how to adjust weights to minimize the overall error.

- **Cholesky Decomposition**: A numerically stable method to work with the Hessian inverse without explicit inversion, crucial for large models.

- **Algorithm Efficiency**: GPTQ's design reduces computational complexity, making it practical for models with billions of parameters.

By understanding the mathematical foundations and practical implementations, we can appreciate the advancements GPTQ brings to the field of neural network quantization.

---

**References**:

- Frantar, E., & Alistarh, D. (2022). Optimal Brain Quantization. *Proceedings of the International Conference on Learning Representations (ICLR)*.
- Nagel, M., Van Baalen, M., Blankevoort, T., & Welling, M. (2020). Up or Down? Adaptive Rounding for Post-Training Quantization. *Proceedings of the International Conference on Machine Learning (ICML)*.