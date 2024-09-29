---
title: 'Parameter Efficient Fine-tuning of LLMs (PEFT)'
date: 2024-09-11
draft: false
---

### Motivation for PEFT

Consider a company like character.ai, which provides different personas for users. For example, you can talk to a chat bot that mimics Elon Musk and ask, "Why did you buy Twitter?" The model responds as Elon Musk would.

Now there are primarily three approaches to solving this:
1. Context-based approach: Take an LLM and provide it with extensive data about the persona (e.g., Elon Musk's interviews and tweets) as context, and then tag on your question. This method is not the most elegant or efficient way to approach the problem and may struggle with persona consistency. Another issue is of the context length, LLMs have a limited context length and we might have more data that can fit into the context length of the model.

2. Full Fine-Tuning: Fine-tune your pre-trained model, updating all the parameters of the model. While full fine-tuning can yield accurate results,but it is a very expensive task. The more concerning issue is scalability and serving the model, It would be an enormous and wasteful undertaking for a company to store hundreds of different finetuned models for all possible personas.

3. This is where technique like PFET fit in. Instead of updating all the model's parameters, PFET keeps the majority of the pre-trained parameters fixed and only adjusts a small subset of the parameters or adds a few new ones. This approach significantly reduces the computational cost and storage requirements, making it feasible to scale and serve models for numerous tasks or personas efficiently.

### Categorization of PFET Algorithms

This is an acitive area of research in NLP right now, There are dozens of excellent Papers comming out every year, on various PFET techniques with each with its unique name (to stand out, i guess?), So it is important to have a framework in place to understand how these techniques fit into the broader landscape, so as not to get overwelmed by these different techniques.

PFET can be broadly classified into four:

1. **Additive Fine-tuning:** Modifies the model architecture by adding new trainable modules or parameters.

2. **Selective Fine-tuning:** Makes only a subset of the model's parameters trainable during the fine-tuning process.

3. **Reparameterized Fine-tuning:** Constructs a low-dimensional reparameterization of the original model parameters for training, then transforms it back for inference.

4. **Hybrid Fine-tuning:** Combines advantages from different PEFT methods to create a unified PEFT model.

<div style="text-align: center;">
  <img src="/images/PEFT/survey_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

### Background

The topics of PFET are not that mathemaically complex, or ...
But understanding the minute details of LLMs is quite essential so i highly recommed reading the GPT2 architecture paper 

### Additive Finetuning

The basic idea in additive finetuning is to incorporate some more params into the LLM, Only these params are updated rest, the original params are kept frozen. So some obvious issuse with this approach is that it adds more params to an already large model, therby reducing infrence speeds.

Differnt methods propose differnt ways of implementing this, so the implementations can get a bit complicated.

Lest start with the simplest of Additive finetunign methods, 

#### Prompt Tuning

##### Prompt tuning image from paper

<div style="text-align: center;">
  <img src="/images/PEFT/PEFT_prompt_tuning_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

Prompt tuning involves adding a few special embeddings, or "prompt vectors," at the beginning of the input sequence. These vectors are not associated with any specific words in the model's vocabulary; instead, they are learnable parameters designed to guide the model's behavior for a particular task. For example, rather than manually crafting a prompt for sentiment analysis, the model learns optimal prompt vectors that maximize performance on the downstream task.

### Mechanism of Prompt Tuning

1. **Initialization**: The prompt vectors are randomly initialized and have the same dimensionality as the model's word embeddings.
2. **Embedding Sequence**: The input sequence for the model includes these special embeddings followed by the original input tokens.
3. **Training**: During training, only the prompt vectors are updated, while the rest of the pre-trained model remains unchanged. This minimizes the number of parameters that need to be fine-tuned, making the process computationally efficient.
4. **Inference**: For inference, the learned prompt vectors are prepended to the input sequence, allowing the model to perform the task effectively.

### Advantages of Prompt Tuning

1. **Parameter Efficiency**: Only a small number of parameters (the prompt vectors) need to be updated, drastically reducing storage requirements and making it easier to share models. Instead of sharing an entire 11-billion-parameter model, one can share just a few prompt vectors.
2. **Batch Processing**: Prompt tuning enables the same base model to handle multiple tasks simultaneously. By including task-specific prompt vectors in the input, a single batch can contain examples from different tasks, streamlining the processing and improving efficiency.

Prompt-Tunig vs Fine Tuing vs Prompting

The blue line represents prompting, where natural language prompts are used without further training. The red and orange lines depict model fine-tuning, where the entire model is fine-tuned on downstream datasets, yielding the highest performance. The green line indicates prompt tuning.
Key Observations
1.	Smaller Models: Prompt tuning underperforms compared to full fine-tuning for smaller models. However, it still provides a significant improvement over plain prompting.
2.	Larger Models: The performance gap between prompt tuning and full model fine-tuning diminishes as model size increases. With large models, prompt tuning achieves comparable results to full fine-tuning, making it an efficient alternative.

<div style="text-align: center;">
  <img src="/images/PEFT/PEFT_promp_tuning_2.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/PEFT_prefix_tuning_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>


<div style="text-align: center;">
  <img src="/images/PEFT/PFET_prefix_tuning_2.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/adapter_fusion_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/adapter_fusion_2.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/dora_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/ia3.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/ia3_2.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/llm_adapter.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/lora_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/lora_2.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/qlora_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/survey_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/survey_2.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/survey_3.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/survey_4.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/survey_5.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/vera_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/PEFT/vera_2.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>