---
title: 'Multi_GPU_Training [Draft]'
date: 2024-10-11
draft: false
comments: false
---
In this post, i wont be discussing code implementations, my goal is to cover the foundational concepts related to multi-GPU Training of Massive llms, 

as stated in my post on Qunatization, you would need a cluster of gpus just to get up and running with the finetuning of of even small llms like the llama 7B models.

The topics i would like to cover are as follows

1. DDP (Distributed Data Parallel)
2. Tensor Model parallelism
3. Pipeline model parallelism
4. Memory efficient pipeline parallelism


Lest start Multi GPU Training 

start with DDP or distributed data parallel.

## DDP (Distributed Data Parallel)

conceptually, ddp is quite simple, most of the effort of ddp lies in making efficient in actual production, dealing wiht race conditions etc...

## Pipeline model parallelism

In data parallelism we say how we can fit multiple copies of the same model on different GPUs, but now we consider the more common scenarion of the model not being able to fit on a single gpu, 
There are primarily two ways we can tackle this problem pipeline parallelism is the more intuitive one, so lets start with that.

The basic idea of pipeline parallelism is quite simple, if your model dosent fit on a single GPU, slice up the different layers and put them across multiple gpus, so each gpu takes input as the output of the previous partition as input, but the problem here is obvious, you cant rent a h100 gpu cluster for 8 bucks an hour and have this bad gpu utilization, so here are some techniques that make model parallelism efficient

1. **PipeDream: Generalized Pipeline Parallelism for DNN Training**

**Pipeline Parallelism in Deep Learning Training: An In-Depth Explanation Inspired by the PipeDream Paper**

---

Pipeline parallelism is a technique used to accelerate the training of deep neural networks (DNNs) by partitioning the computation graph across multiple devices, such as GPUs. The PipeDream paper introduces a novel approach to pipeline parallelism that addresses the limitations of traditional data and model parallelism methods. Below is a detailed explanation of pipeline parallelism as described in the PipeDream paper.

### **Background**

Traditional parallelization strategies for training DNNs include:

1. **Data Parallelism (DP):** Distributes different data samples (minibatches) across multiple GPUs, each with a complete copy of the model. After computing gradients, the GPUs synchronize to update the model parameters.
   
2. **Model Parallelism (MP):** Splits the model itself across multiple GPUs. Each GPU holds a portion of the model and processes the same data sample sequentially through the different parts.

While these methods have their advantages, they also have limitations, especially when scaling to large models or a high number of GPUs. Pipeline parallelism aims to overcome these limitations by combining aspects of both data and model parallelism.

### **What is Pipeline Parallelism?**

Pipeline parallelism involves dividing the layers of a DNN into sequential stages and assigning each stage to a different GPU. Each GPU is responsible for the forward and backward computations of its assigned layers. By injecting multiple minibatches into the pipeline, all GPUs can work simultaneously, processing different minibatches at different stages.

### **How Pipeline Parallelism Works**

1. **Partitioning the Model:**
   - The DNN is divided into several stages, each containing a consecutive set of layers.
   - Each stage is assigned to a separate GPU.

2. **Injecting Minibatches:**
   - Multiple minibatches are introduced into the pipeline sequentially.
   - As one GPU completes the forward pass for a minibatch, it sends the output activations to the next GPU and starts processing the next minibatch.

3. **Forward and Backward Passes:**
   - The last stage (GPU) starts the backward pass immediately after completing the forward pass for a minibatch.
   - Each GPU performs the backward pass for its stage and sends the gradients to the previous GPU while starting computations for the next minibatch.

4. **Asynchronous Communication:**
   - Communication of activations and gradients between GPUs is done asynchronously.
   - This allows for overlapping computation and communication, improving overall efficiency.

### **Advantages of Pipeline Parallelism**

1. **Reduced Communication Overhead:**
   - Communication is limited to adjacent GPUs, transferring only the necessary activations and gradients.
   - This is more efficient than DP, which requires global synchronization and communication of all model parameters.

2. **Improved Resource Utilization:**
   - By keeping multiple minibatches in flight, all GPUs remain active, reducing idle time.
   - Overlapping computation and communication maximizes hardware utilization.

### **Challenges and Solutions in PipeDream**

The PipeDream paper identifies three main challenges in implementing effective pipeline parallelism and proposes solutions for each.

#### **Challenge 1: Work Partitioning**

**Problem:**
- Uneven computational workloads across stages can lead to pipeline bubbles, where some GPUs are idle waiting for others to complete.
- Excessive communication between GPUs can reduce throughput.

**Solution:**
- **Automated Partitioning Algorithm:**
  - Profiles the DNN to estimate computation times and output sizes for each layer.
  - Uses dynamic programming to partition layers into stages such that each stage has a balanced computational load.
  - Takes into account hardware topology and communication bandwidth to minimize communication overhead.
  - Allows for stage replication (using data parallelism within a stage) when perfect load balancing isn't possible with simple partitioning.

**Process:**
1. **Profiling:**
   - Measure computation times (forward and backward passes) and activation sizes for each layer.
2. **Optimization:**
   - Solve a dynamic programming problem to find the optimal partitioning that balances the workload and minimizes communication.
   - Consider replication factors for stages to further balance the pipeline.

#### **Challenge 2: Work Scheduling**

**Problem:**
- Deciding whether a GPU should perform a forward or backward pass at any given time.
- Routing minibatches correctly when stages are replicated.

**Solution:**
- **One-Forward-One-Backward (1F1B) Scheduling:**
  - Each GPU alternates between performing a forward pass for one minibatch and a backward pass for another minibatch.
  - This schedule ensures that all GPUs are continuously utilized.

- **Deterministic Round-Robin Load Balancing (1F1B-RR):**
  - When stages are replicated, minibatches are assigned to replicas in a round-robin fashion based on their IDs.
  - Ensures that each minibatch is processed by the same GPU for both forward and backward passes within a stage.

**Process:**
1. **Startup Phase:**
   - The pipeline is filled with an optimal number of minibatches to reach steady state.
2. **Steady State:**
   - GPUs follow the 1F1B schedule, maintaining a balance between forward and backward computations.

#### **Challenge 3: Effective Learning**

**Problem:**
- Inconsistency in parameter versions used during forward and backward passes can lead to invalid gradients and hinder convergence.
- Since parameters are updated asynchronously across stages, a minibatch might use different parameter versions in its forward and backward passes.

**Solution:**
- **Weight Stashing:**
  - Store (stash) the parameters used during the forward pass of each minibatch.
  - Use the same stashed parameters during the backward pass to compute gradients.
  - Ensures that gradients are computed consistently with the parameters used in the forward pass.

- **Vertical Sync (Optional):**
  - Coordinates the use of parameter versions across stages.
  - Each minibatch uses the same parameter version for both forward and backward passes across all stages.
  - Involves more coordination and storage but provides consistency similar to synchronous data parallelism.

**Process:**
1. **During Forward Pass:**
   - Use the latest available parameters.
   - Stash the parameters for each minibatch.
2. **During Backward Pass:**
   - Retrieve the stashed parameters corresponding to the minibatch.
   - Compute gradients and update parameters accordingly.

### **Understanding Staleness and Consistency**

- **Staleness:**
  - Refers to the difference in parameter versions used when computing gradients.
  - Weight stashing reduces staleness within a stage but doesn't eliminate it across stages.
  
- **Consistency Models:**
  - **Without Weight Stashing:** Parameters may be inconsistent, leading to invalid gradients.
  - **With Weight Stashing:** Consistent within a stage; some staleness across stages.
  - **With Vertical Sync:** Consistent across all stages for each minibatch; mimics synchronous training.

### **Memory Considerations**

- **Memory Overhead:**
  - Weight stashing increases memory usage since parameters need to be stored for each in-flight minibatch.
  - However, the per-GPU memory usage remains comparable to data parallelism.

- **Optimization Techniques:**
  - **Activation Recomputation:** Discard activations after forward pass and recompute them during backward pass to save memory.
  - **Gradient Accumulation:** Aggregate gradients over multiple minibatches before updating parameters.

### **Implementation Highlights**

- **PipeDream Runtime:**
  - Manages device memory, schedules tasks, and handles communication between GPUs.
  - Integrates with deep learning frameworks like PyTorch.

- **Communication Backend:**
  - Uses efficient communication libraries (e.g., Gloo, NCCL) for transferring activations and gradients.

- **Checkpointing:**
  - Supports periodic saving of model parameters for fault tolerance.
  - Each stage checkpoints independently, reducing coordination overhead.

### **Benefits of PipeDream's Pipeline Parallelism**

- **Scalability:**
  - Enables training of larger models that don't fit into the memory of a single GPU.
  - Efficiently utilizes multiple GPUs without incurring excessive communication overhead.

- **Throughput Improvement:**
  - By keeping all GPUs busy and overlapping computation with communication, PipeDream achieves higher throughput compared to traditional methods.

- **Flexibility:**
  - Can be combined with data parallelism within stages (hybrid parallelism) for further scalability.

### **Conclusion**

Pipeline parallelism, as implemented in the PipeDream paper, presents an effective method for scaling DNN training across multiple GPUs. By carefully partitioning the model, scheduling work to maximize GPU utilization, and ensuring consistent parameter usage through weight stashing, PipeDream overcomes the challenges associated with pipeline parallelism. This approach leads to significant improvements in training throughput while maintaining model convergence, making it a valuable technique for training large-scale deep learning models.

8:37
parallelism uh we split the layers or operators in the model over multiple devices uh and then we also will split
8:45
each batch of inputs into smaller micro batches and then paralyze execution across these micro
8:52
batches to be very concrete uh let's look at this visually um so this is a
8:57
model uh that we're splitting over four devices uh so let's say that if the if the model
9:06
has eight Transformer layers uh what we're going to do is we're going to assign the first two Transformer layers
9:12
to the first device the next two to the second device and so on now in order to perform a single
9:18
forward and backward path through the model we're going to need to take a single input pass it through device one
9:26
device one performs its computation uh represented by this blue box computes an
9:32
what what we call an output activation and then this output activation needs to be communicated to the next device uh uh
9:40
and and and and and the second device can start it uh it its computation until it's receive this activation from the
9:47
the first device um and and so what that means is that there is this sequential
9:53
data dependency across each of these devices um and lots of these devices are
9:58
idle um in in particular at any point in time only one device is
10:04
active um and so so very quickly you can see that uh this scheme has uh pretty uh
10:11
poor utilization and low throughput so instead what we can do is we can take this input batch a um and
10:19
split it into smaller micro batches uh let's say that this this uh input batch
10:26
a has has four inputs in it um what we can do is we can split that um input of
10:32
uh input batch of four into four micro batches of size one and then pipeline execution across um those micro
10:41
batches um in particular um this is this is what this looks like um we note now
10:46
that we only have sequential uh sequential data dependencies um between
10:53
uh devices for a given microbatch um in other words um device 2 now only needs
10:59
to wait on device one for um uh this output activation of microbatch A1
11:06
before it starts computation so no longer do you have to wait for all four for for device one to complete uh
11:12
computation for all four um input samples in in in in in this patch um
11:18
instead we can just um we can immediately start uh processing on device 2 as soon as just an a single
11:26
input's uh worth of computation is is completed on on device
11:35
one after we complete uh uh computation for all of these forward and and
11:42
backward passes for these four um uh micro batches uh then we can step the
11:47
the optimizer uh which is basically around here um and then we can update
11:54
the the weights and move on to the next training iteration
12:00
it's easy to see from from from these figures that uh this is much more efficient um compared to the the naive
12:07
case where um we only have a single batch uh but there are still some idle
12:14
periods we haven't completely eliminated um these idle periods from
12:20
from from these timelines um we call um the periods of of time that each device
12:27
is is Idle um uh at the start and end of a training iteration the pipeline flush
12:33
um and and these are basically fundamental right um uh basically the pipeline flush is the time that devices
12:39
need to wait for inputs to actually flow through the the pipeline um and then
12:45
subsequently get drained
12:52
out so to summarize with pipeline model parallelism we need to perform uh point-to-point communication between
12:58
between consecutive pipeline stages uh and we have these pipeline Bubbles at the start and end of every
13:05
batch we can actually exactly quantify how much time is spent in the pipeline bubble uh it's actually going to be
13:11
equal to P minus one microb batches worth of forward and backward passes uh
13:16
where p is the number of pipeline stages um so in the previous figure uh the uh
13:23
the number of pipeline stages was four um and the size of the pipeline bubble was three micro batches worth of forward
13:31
and and backward pass


------------

# References

[1] https://huggingface.co/blog/bloom-megatron-deepspeed#bf16optimizer

[2] https://lightning.ai/blog/doubling-neural-network-finetuning-efficiency-with-16-bit-precision-techniques/