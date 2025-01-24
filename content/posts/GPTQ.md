+++
title = 'Quantization in LLMS Part 2: GPTQ [Draft]'
date = 2024-08-01T09:51:17-04:00
draft = false
math = true
+++

**Introduction**

Quantization is a crucial technique in deep learning that reduces the memory footprint and computational requirements of neural networks by representing weights and activations with lower-precision numerical formats. This is particularly important when deploying large models on devices with limited resources. However, quantizing a neural network without significantly degrading its performance is challenging.

The GPTQ (Gradient Post-Training Quantization) algorithm is a method designed to efficiently quantize large-scale neural networks, such as those used in natural language processing, while maintaining high accuracy. GPTQ builds upon previous methods like Optimal Brain Quantization (OBQ) but introduces significant modifications to make it scalable to models with billions of parameters.

In this explanation, we will delve into the mathematical foundations of GPTQ, starting with the Optimal Brain Sergeon. xplai..... how it leverages the Hessian matrix and its inverse, discuss the role of the Cholesky decomposition, and provide a detailed walkthrough of the algorithm. We will also include examples to illustrate key concepts.

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
- $ \| \cdot \|_2 $: Frobenius norm, summing over all elements. (similar to enclidian norm for vectors)

**Goal**: Find $ \mathbf{W}_c $ that minimizes the output difference caused by quantization.

---

One of the earlier methods proposed to solve this problem is the OBS (Optimal Brain Sergeon) approach. The GPTQ algorithm is base on the OBQ(Optimal Brain Quantization), which in turn is based on the OBS(Optimal Brain Sergeon) paper, and most of the Equantions prsent in the GPTQ papers are taken from the OBS Paper, so if you want to understand how the weights are quantized and where do these complex formulas come from, you need to First understand the OBS Paper.

The meat of the OBS Paper is a Problem of Constrained Optimization, so i will provide a breifoverview of COnstrained opimization and some relevant formulas for understanding the topics ahead.
(This post assumes that you are familiar wiht terms like gradient and Hessians, and this is not the first time you are hearing the term optimaization and Taylor series) 

## A Quick Refresher on Constrained Optimaizaiton

### Problem Setup:

You have two things in a constrained optimization problem:
1. **Objective function** $ f(x) $ — the function you want to maximize or minimize.
2. **Constraint** $ g(x) = 0 $ — an equation that defines the condition your solution must satisfy.

Mathematically, the goal is to:
$$
\text{Minimize } f(x) \quad \text{subject to } g(x) = 0
$$

### Lagrange Multipliers:
To solve this problem, we introduce a new variable, called a **Lagrange multiplier**, denoted as $ \lambda $. The idea is to combine the objective function and the constraint into a single equation, called the **Lagrangian**:

$$
\mathcal{L}(x, \lambda) = f(x) + \lambda g(x)
$$

- Here, $ f(x) $ is the objective function.
- $ g(x) = 0 $ is the constraint.
- $ \lambda $ is the Lagrange multiplier, which "enforces" the constraint.


---

### Why Does This Work? (Super Optional: Read if Intrested....)

The logic behind this is that at the optimal point under the constraint, the gradient of the objective function $ \nabla f(x) $ must be aligned with the gradient of the constraint $ \nabla g(x) $.

#### **Why Are the Gradients Aligned?**

#### **1. The Gradient and Direction of Steepest Ascent**

- **Gradient of a Function ($ \nabla f(x) $)**: Points in the direction of the steepest increase of the function $ f(x) $ at point $ x $.

- **Constraint Surface ($ g(x) = 0 $)**: Defines a surface (or curve) in the space of $ x $. Any movement along this surface keeps $ g(x) $ constant.

#### **2. Feasible Directions and Tangent Space**

- **Feasible Directions**: Directions in which you can move without violating the constraint $ g(x) = 0 $. These directions lie in the **tangent space** of the constraint surface at point $ x $.

- **Tangent Space**: The set of all vectors $ d $ such that $ \nabla g(x)^\top d = 0 $. This means moving a small amount $ \epsilon d $ keeps you on the constraint surface to first order.

#### **3. Optimality Condition Under Constraint**

- At the **optimal point**, you cannot find a feasible direction $ d $ that will further decrease $ f(x) $.

- **Mathematically**: The directional derivative of $ f $ in any feasible direction $ d $ must be zero:
  $$
  \nabla f(x)^\top d = 0 \quad \text{for all } d \text{ such that } \nabla g(x)^\top d = 0
  $$

#### **4. Gradients Must Be Parallel**

- The only way for $ \nabla f(x)^\top d = 0 $ for all feasible $ d $ is if $ \nabla f(x) $ is a **linear combination** of $ \nabla g(x) $.

- **Conclusion**: There exists a scalar $ \lambda $ such that:
  $$
  \nabla f(x) = -\lambda \nabla g(x)
  $$
  This means $ \nabla f(x) $ and $ \nabla g(x) $ are **aligned** (parallel or antiparallel).

---

### Steps to Solve:

1. **Form the Lagrangian**: As mentioned earlier, combine the objective and constraint:
   $$
   \mathcal{L}(x, \lambda) = f(x) + \lambda g(x)
   $$

2. **Take Partial Derivatives**:
   - Take the derivative of $ \mathcal{L} $ with respect to $ x $ (the variables in your objective function).
   - Take the derivative of $ \mathcal{L} $ with respect to $ \lambda $.

3. **Set Derivatives to Zero**: 
   $$
   \frac{\partial \mathcal{L}}{\partial x} = 0 \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial \lambda} = 0
   $$
   The first condition ensures that the gradients of $ f(x) $ and $ g(x) $ are aligned, and the second condition ensures that the constraint $ g(x) = 0 $ is satisfied.

4. **Solve the System of Equations**: 
   You now have a system of equations involving $ x $ and $ \lambda $. Solve this system to find the optimal values of $ x $ (and $ \lambda $, though you usually don’t care about its value directly).


### A quick review of Taylor Series

A function $f(x + \Delta {x})$ may be Taylor expanded about a base point $x$ (if $f$ is differentiable at $x$):

$$
f(x + \Delta {x}) = f(x) + \frac{d f(x)}{d {x}} \Delta {x} + \frac{d^2 f(x)}{d {x}^2} \frac{\Delta {x}^2}{2!} + \dots
$$

We don't deal with scalars (anymore!)

$$
f(\mathbf{x} + \Delta \mathbf{x}) = f(\mathbf{x}) + \nabla f(\mathbf{x}) \cdot \Delta \mathbf{x} + \frac{1}{2!} \Delta \mathbf{x}^T H(f(\mathbf{x})) \Delta \mathbf{x} + \dots
$$

- $\nabla f(\mathbf{x})$ = gradient of $f$ at $\mathbf{x}$, a vector of partial derivatives

$$
\nabla f(\mathbf{x}) = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)
$$

- $H(f(\mathbf{x}))$ = second-order partial derivatives of $f$ at $\mathbf{x}$, with elements $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$

$$
\Delta \mathbf{x}^T H(f(\mathbf{x})) \Delta \mathbf{x}
$$
is a quadratic form, this term computes how much the function curves in the direction of $\Delta \mathbf{x}$, weighted by the second derivatives (Hessian).

Phew, we are done with our Background so lets Begin

## OBS (Optimnal Brain Surgery)

### Overview
OBS is a method for pruning neural networks by removing weights while minimizing the increase in the loss fuction. It uses secone order (Hessian ) information to identify which weights can be pruned with minimal impact on performance.

### Problem Context
In a neural network, each weight $ w_p $ contributes to the overall performance (or loss) of the model. If you want to **remove** a weight (i.e., prune it), there is a chance that this removal will negatively affect the network's performance, causing the loss function to increase.

To avoid large increases in the loss function, OBS prunes weights in a **careful, optimized way**. The goal is to find which weight(s) can be removed and adjust the other weights such that the **impact on the loss function is minimal**.

### Optimization Problem
This all sounds great, but how is this an optimization problem?
When you prune a weight, it affects the entire neural network. Simply setting a weight to zero will not work well unless you **adjust the other weights** in the network to compensate for the change. This adjustment of weights forms an optimization problem:

- **Objective:** Minimize the **increase in the loss function** caused by pruning.
- **Constraint:** Set the specific weight $ w_p $ to zero (i.e., prune it).

This optimization problem helps find the best way to modify the remaining weights while ensuring the selected weight is removed.


### Mathematical Formulation
The loss function $ L(\mathbf{w}) $ of a neural network depends on the weights $ \mathbf{w} = [w_1, w_2, \dots, w_n] $. When we prune a weight, we want to minimize the change in the loss function, specifically:

$$
\Delta L = L(\mathbf{w} + \delta \mathbf{w}) - L(\mathbf{w})
$$

$
\mathbf{w} \in \mathbb{R}^n \quad \text{original set of weights}
$

$
\delta\mathbf{w} \in \mathbb{R}^n \quad \text{change applied to the weights}
$ (In case of pruning a specific weight $w_p$, $\delta w_p$ would be a negative value)$ \delta w_p = -w_p $

2nd order Taylor Series Expansion

$$
L(\mathbf{w} + \delta \mathbf{w}) = L(\mathbf{w}) + \nabla L(\mathbf{w})^T \delta \mathbf{w} + \frac{1}{2} \delta \mathbf{w}^T H \delta \mathbf{w}
$$

**Neglecting the gradient!!!**, this is someting that we genrally dont do, (we generally ignore the first derivative)

In practice, when we are pruning weights, we assume the network is already near a local minimum (already trained), so the gradient $\nabla L(\mathbf{w})$ is close to 0. This simplifies the expression.

$$
\Delta L = L(\mathbf{w}) + \nabla L(\mathbf{w})^T \delta \mathbf{w} + \frac{1}{2} \delta \mathbf{w}^T H \delta \mathbf{w} - L(\mathbf{w})
$$

Therefore,

$$
\Delta L = \frac{1}{2} \delta \mathbf{w}^T H \delta \mathbf{w}
$$
### **Setting up the Optimization Problem:**
The optimization problem becomes:

$$
\min_{\delta \mathbf{w}} \frac{1}{2} \delta \mathbf{w}^\top H \delta \mathbf{w}
$$

Subject to the constraint:

$$
\delta w_p = -w_p
$$

This constraint enforces the pruning of the weight $ w_p $, meaning we want the change in $ w_p $ to be exactly $ -w_p $ so that the new value becomes zero (i.e., $ w_p + \delta w_p = 0 $).

### Solving the optimization problem

### Step-by-Step Derivation:

#### 1. **Formulating the Objective:**
The objective is to **minimize** the increase in the loss function due to weight pruning. 

$$
\min_{\delta \mathbf{w}} \frac{1}{2} \delta \mathbf{w}^\top H \delta \mathbf{w}
$$

Here:
- $ \delta \mathbf{w} $ represents the change in the weights.
- $ H $ is the Hessian matrix, which gives the second-order approximation of how the loss function changes with respect to the weights.

#### 2. **Setting the Constraint:**
When we prune a specific weight $ w_p $, the change in that weight should be exactly $ -w_p $ to set it to zero:

$$
\delta w_p = -w_p
$$

In vector form, this constraint can be written as:

$$
\mathbf{e}_p^T \delta \mathbf{w} + w_p = 0
$$

where $ \mathbf{e}_p $ is the one hot vector corresponding to the weight $ w_q $.

This ensures that the weight $ w_p $ is fully pruned.

#### 3. **Setting up the Lagrangian:**
To handle this constraint, we introduce a **Lagrange multiplier** $ \lambda $. The Lagrangian $ \mathcal{L} $ combines the objective function and the constraint:

$$
\mathcal{L}(\delta \mathbf{w}, \lambda) = \frac{1}{2} \delta \mathbf{w}^\top H \delta \mathbf{w} + \lambda (\mathbf{e}_p^T \delta \mathbf{w} + w_p)
$$

Where:
- The term $ \lambda (\mathbf{e}_p^T \delta \mathbf{w} + w_p) $ enforces the constraint that $ \delta w_p = -w_p $.
- $ \lambda $ is the Lagrange multiplier.

#### 4. **Taking the Derivative of the Lagrangian:**
To minimize the Lagrangian, we take the derivative with respect to the change in weights $ \delta \mathbf{w} $ and the Lagrange multiplier $ \lambda $.

##### a. **Derivative w.r.t. $ \delta \mathbf{w} $:**

$$
\frac{\partial \mathcal{L}}{\partial \delta \mathbf{w}} = H \delta \mathbf{w} + \lambda \mathbf{e}_p = 0
$$

Here:
- $ e_p $ is a vector with a 1 in the $ p $-th position (corresponding to $ w_p $) and 0 elsewhere, as the constraint only applies to the $ p $-th weight.
- Solving this gives:

$$
H \delta \mathbf{w} = -\lambda \mathbf{e}_p
$$

##### b. **Derivative w.r.t. $ \lambda $:**

$$
\frac{\partial \mathcal{L}}{\partial \lambda} = \delta w_p + w_p = 0
$$

This gives us the constraint:

$$
\delta w_p = -w_p
$$

#### 5. **Solving for $ \delta \mathbf{w} $:**
From the equation $ H \delta \mathbf{w} = -\frac{\lambda}{2} e_p $, we can solve for $ \delta \mathbf{w} $ by multiplying both sides by the inverse of $ H $:

$$
\delta \mathbf{w} = -\frac{\lambda}{2} H^{-1} e_p
$$

The vector $ H^{-1} e_p $ represents the $ p $-th column of the inverse of the Hessian matrix $ H $.

#### 6. **Solving for $ \lambda $:**
We now use the constraint $ \delta w_p = -w_p $ to solve for $ \lambda $.

From $ \delta w_p = e_p^\top \delta \mathbf{w} = -w_p $, we substitute $ \delta \mathbf{w} $:

$$
e_p^\top \left( -\frac{\lambda}{2} H^{-1} e_p \right) = -w_p
$$

This simplifies to:

$$
-\frac{\lambda}{2} [H^{-1}]_{pp} = -w_p
$$

Solving for $ \lambda $:

$$
\lambda = \frac{2 w_p}{[H^{-1}]_{pp}}
$$

#### 7. **Substitute $ \lambda $ Back into $ \delta \mathbf{w} $:**
Now, substitute $ \lambda $ back into the expression for $ \delta \mathbf{w} $:

$$
\delta \mathbf{w} = -\frac{w_p}{[H^{-1}]_{pp}} H^{-1} e_p
$$

This equation gives the **update to the remaining weights** after pruning $ w_p $.



### Finding $ \delta w $
Now, substitute $ \lambda $ back into the expression for $ \delta w $:

$$
\delta w = -\frac{w_q}{e_q^T H^{-1} e_q} H^{-1} e_q
$$

This is the optimal weight change that minimizes the increase in error while setting $ w_q $ to zero.

### Final Change in Error ($ L_q $)
Now, we compute the resulting change in error $ \delta E $ by substituting $ \delta w $ into the second-order Taylor expansion of the error function:

$$
\delta E = \frac{1}{2} \delta w^T H \delta w
$$

Substitute $ \delta w = -\frac{w_q}{e_q^T H^{-1} e_q} H^{-1} e_q $ into this:

$$
\delta E = \frac{1}{2} \left(-\frac{w_q}{e_q^T H^{-1} e_q} H^{-1} e_q \right)^T H \left(-\frac{w_q}{e_q^T H^{-1} e_q} H^{-1} e_q \right)
$$

Simplifying:

$$
\delta E = \frac{1}{2} \frac{w_q^2}{(e_q^T H^{-1} e_q)^2} \left( e_q^T H^{-1} H H^{-1} e_q \right)
$$

Since $ H^{-1} H = I $, this becomes:

$$
\delta E = \frac{1}{2} \frac{w_q^2}{e_q^T H^{-1} e_q}
$$

This is the final expression for the change in error due to setting $ w_q $ to zero, which we denote as $ L_q $:

$$
L_q = \frac{1}{2} \frac{w_q^2}{(H^{-1})_{qq}}
$$

where $ (H^{-1})_{qq} = e_q^T H^{-1} e_q $ is the diagonal element of the inverse Hessian matrix corresponding to weight $ w_q $. This final equation represents the "saliency" or importance of weight $ w_q $ in the network.



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



