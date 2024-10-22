---
title: 'RLHF for Dummies'
date: 2024-10-12
draft: true
comments: false
---

1. Motivation
2. SFT
3. RLHF Mathematics


Linked IN, 
I am a huge fan of Reinforcement Learning, I believe it lead to the LLM revolution that we are living in today, without rlhf and Instruction Tuning LLM would still be giveing answers like below, so lets take a journey of RLHF.

This post is not going to be about Instruction Tuning, (The field of SFT/Instruction tuning is moving rapidly and i would like to do a dedicated post on that.)

Now with the disclamer out of the way, lets step into the merky waters of RLHF.

Will incorporate the NYU lecture by the open ai guy later.
# First from UMASS 

Before we begin what SFT and what instruction tuning and RLHF do, lets first take a look at the difference between just a pretrained model and an SFT + RLHF model is, (i feel people who think, language modesl are smart and have great abilities, i highly recomened use just a pretrained Language model, you will appreciate how dumb they are, almost 95% of the training is done, only 5% of how to answer a questions such that humans will like is taught during sft and rlhf, so ....) The beow image is from the GPT 3 family of models, in which you can clearly see, 


<div style="text-align: center;">
  <img src="/images/rlhf/rlhf_instruct_gpt.png" alt="Comparison of 32-bit, 16-bit, and bfloat16 floating-point formats." style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Comparison of 32-bit, 16-bit, and bfloat16 floating-point formats.
  <a href="https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html" style="color: rgba(0, 0, 0, 0.6);">(Maxime Labonne)</a>
</p>
</div>


0:39
an example of what we are talking about here so this
0:46
um prompt here what is the purpose of the list C in the code below and you have this snippet of code here
0:54
um if you can see this and so last time we talked about these huge scale language
0:59
models and how they're so capable of doing things but this is a completion of
1:04
this prefix from gpt3 um as you can see it uh decides to
1:11
generate multiple choices uh and basically trying to convert this this
1:17
prefix into uh some kind of exam question right this is not the intended
1:23
Behavior the person inputting this prefix into the model wanted an actual answer not multiple choices for someone
1:31
else to answer right and so uh you see this much more broadly
1:36
with many different types of prompts so for example you might want your model to
1:42
you know not output harmful text in response to a particular prompt or not output misleading
1:49
um text or non-factual text or many different kinds of undesirable properties but as we know these models
1:56
are trained on to predict the next word on the internet and there's many undesirable things in there and also the
2:03
model learns weird things like this where it thinks that a likely continuation to this prefix is just to
2:10
make a multiple choice question which is clearly not what we want so um the purpose of this is just to
2:18
demonstrate that next word prediction on a large data set can only get you so far
2:23
because you're learning like what is most likely based on what I've seen in
2:29
my data but maybe we want to change the behavior of the language model based on you know what humans actually find
2:36
useful so after applying some of the techniques that we'll talk about in this class we see that the generated sample
2:46
from this model is much better right it's actually explaining what this list C is doing instead of something
2:54
completely useless like this so um this is kind of the high level motivation here


## RLHF Aligning Language modesl to follow human prefrence.

# The real stuff

spring 2023 Umass Lecture

### Motivation

### Instruction Finetuning

Limitations of Instruction Finetuning.

1. Expensive to collect ground truth data
2. open ended tasks have no right answer

## RLHF

Alignment of LLMs with human intent

1. start with a large pre trained model
2. Instruction Finetuning

