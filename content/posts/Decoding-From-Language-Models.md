---
title: "Deconding From Language Models"
date: 2024-09-11
draft: true
---

## A quick refresher on Autoregressive text generation

<div style="text-align: center;">
  <img src="/images/Decoding/decoding_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

Autoregressive language models generate text through a sequential process of predicting one token at a time. The model takes a sequence of tokens $ \lbrace y \rbrace _{<t} $ as input  and outputs a new token $ \hat{y_t} $. This process repeats iteratively, with each newly generated token becoming part of the input for the subsequent prediction.

At each time step $ t $, the model computes a vector of scores $ \mathbf{S} \in \mathbb{R}^V $, where $ V $ is the size of the vocabulary. For instance, in GPT-2, the vocabulary consists of 50,257 tokens.

$$
S = f(\lbrace y \rbrace _{<t} )
$$
<div style="text-align: center;">
where, $ f $ is the Language model.
</div>


The model then applies the softmax function to these scores to obtain a probability distribution over the vocabulary, representing the likelihood of each token being the next word given the preceding context.


$$
P(y_t = w \mid \{ y_{<t} \}) = \frac{\exp(S_w)}{\sum_{w' \in V} \exp(S_{w'})}
$$

Here, $ S_w $ is the score of token $ w $, and the denominator is the sum of the exponentiated scores of all tokens in the vocabulary. Once we have this probability distribution, we can start decoding.

In the upcoming sections, we will explore different decoding methods, starting with greedy decoding, to understand how models convert these probabilities into coherent text.

## Greedy Decoding

Greedy decoding is the most straightforward approach. Much like other classification tasks, where we take the argmax as the output, greedy decoding selects the token with the highest probability at each time step.

$$ \hat{y_t} = \arg\max_{w \in V} P(y_t = w \mid \lbrace y \rbrace _{<t}) $$

Below is an example of unconditional text generation. The model starts with a special \<START> token, which can also be something like a newline character '\n', and iteratively selects the most probable token at each step until it reaches the \<END> token or some external maximum sequence length is met.

<div style="text-align: center;">
  <img src="/images/Decoding/greedy.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

The issue with greedy decoding is that once a decision is made you can't go back and change the decision, like there might have been a word like “a” or “the” If we could have taken it differently, we could eventually end up with a higher probability statement overall.

There is a simple way to get around this limitation of only considering one possible hypothesis.

### Beam Search

The fundamental idea of beam search is to explore multiple hypotheses simultaneously rather than focusing on a single one. It keeps track of the $ k $ most probable partial sequences at each decoder step instead of just the top one. $ k $ is a parameter known as the beam width. This approach allows the algorithm to explore multiple hypotheses simultaneously, which improves the likelihood of finding a globally optimal sequence.

#### Beam Search Algorithm

1. **Starting Beam:**

  At time step $ t = 0 $, we start with the beam containing only the start-of-sequence token, denoted as $ \langle s \rangle $. Each sequence in the beam is paired with its cumulative log probability.

  $$
  \text{Beam}_0 = \lbrace{ \left( \langle s \rangle, \log P(\langle s \rangle) \right) }\rbrace
  $$

  Since $ \langle s \rangle $ is the starting token, we often initialize its log probability to zero:

  $$
  \text{Beam}_0 = \lbrace{ \left( \langle s \rangle, 0 \right) } \rbrace
  $$

2. **Expanding Sequences:**

   For each sequence $ Y_{1:t-1} $ in the beam, we generate all possible extensions by appending each possible next token $ \hat{y_t} $. Each extension forms a new candidate sequence:

   $$
   Y_{1:t} = Y_{1:t-1} \oplus \hat{y_t}
   $$
   
   Here, $ \oplus $ denotes the concatenation of the existing sequence with the new token.
3. **Computing Log Probabilities:**

   For each new candidate sequence $ Y_{1:t} $, we calculate the cumulative log probability. The log probability is used instead of the actual probability to prevent numerical underflow and to turn the product of probabilities into a sum, which is computationally more stable.


   $$
   \log P(Y_{1:t}) = \log P(\hat{y_t} \mid Y_{1:t-1}) + \log P(Y_{1:t-1})
   $$

   This combines the log probability of the new token $ \hat{y_t} $ given the previous sequence $ Y_{1:t-1} $ with the cumulative log probability of the previous sequence.

4. **Generating Candidates:**

   Collect all possible candidates formed by extending each sequence in the current beam with all possible next tokens:

   $$
   \text{Candidates} = \{ (Y_{1:t-1} \oplus \hat{y_t}, \log P(\hat{y_t} \mid Y_{1:t-1}) + \log P(Y_{1:t-1})) \}
   $$

5. **Pruning:**

   From the set of all candidates, select the top $ k $ sequences based on their log probabilities to form the new beam:

   $$
   \text{Beam}_t = \text{top}_k(\text{Candidates})
   $$

   This step ensures that only the $ k $ most probable sequences are kept, and the rest are discarded.

6. **Repeating:**

    Repeat steps 2-5 until a stopping condition is met (e.g., maximum length reached or end token generated).

7. **Final Output:**
  After the iteration stops, we select the sequence with the highest cumulative log probability from the final beam.

  $$
  Y^\ast = \arg\max_{Y \in \text{Beam}_T} \log P(Y)
  $$


Beam Search with an Example of K =2

<div style="text-align: center;">
  <img src="/images/Decoding/beam_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

Get the two most probable tokens, rest are pruned.

<div style="text-align: center;">
  <img src="/images/Decoding/beam_2.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

top 2 generations are “the poor” and “a poor”, these are kept and the reset are pruned

<div style="text-align: center;">
  <img src="/images/Decoding/beam_3.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

The above process is continued till we reach the \<EOS> end of sequence token or reach a limit for max output tokens.

<div style="text-align: center;">
  <img src="/images/Decoding/beam_4.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

Greedy decoding and Beam Seach are some of the most popular approaches for decoding in LLM’s like chat GPT, Llam and Gemini. 

### The issue with these approaches.

<div style="text-align: center;">
  <img src="/images/Decoding/issues_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

Open-ended text generation often leads to repetitive outputs. For example, when generating text about a unicorn trying to speak English, the continuation may initially appear coherent but soon start repeating phrases, like an institution's name, excessively. This repetition happens because the language model assigns increasing probability to repeated sequences, as shown in a plot of the model's probability for the sequence "I don't know." Initially, the probability is regular, but as the phrase repeats, the probability increases, indicating the model is more confident about the repetition.

<div style="text-align: center;">
  <img src="/images/Decoding/issues_2.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

This issue, known as self-amplification, persists even with larger models. For instance, models with 175 billion parameters still suffer from repetition when generating the most likely string. Increasing scale alone does not resolve this problem.
To mitigate repetition, one approach is n-gram blocking, which prevents the same n-gram from appearing twice. For example, if n is set to three, and the text contains "I am happy," the next time "I am" appears, "happy" would be set to zero probability, preventing the repetition of this trigram. However, n-gram blocking has limitations, as it can eliminate necessary repetitions, such as a person's name appearing multiple times.

<div style="text-align: center;">
  <img src="/images/Decoding/issues_3.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

Finally, let's discuss whether generating the most likely string is reasonable for open-ended text generation. The answer is probably no, as it doesn't align well with human patterns. In this graph, the orange curve represents human-generated text, while the blue curve shows machine-generated text using beam search. You can observe that human speech exhibits a lot of uncertainty, evident from the fluctuations in probabilities. For some words, humans are very certain, while for others, they are less sure. In contrast, the model distribution is always very confident, consistently assigning a probability of one to the sequence. This clear mismatch between the two distributions suggests that searching for the most likely string may not be the appropriate decoding objective.


## Sampling Based Decoding

Another way of looking at the decoding problem is that we have a conditional probability distribution over the next token, why not just sample from this distribution?

### Ancestral Sampling

At each time step, we're going to draw a token from the probability distribution according to its relative likelihood. This process involves sampling a token from the distribution $ P(y_t = w \mid \{ y \}_{<t}) $. Essentially, we are trying to sample $\hat{y_t}$ from this distribution.


Ancestral sampling allows us to select any token based on its probability, rather than being restricted to the most probable token.

$$
\hat{y_t} \sim P(y_t = w \mid \{ y \}_{<t})
$$

For example, previously you might have been restricted to selecting "restroom," but with sampling, you might select "bathroom" as well.

<div style="text-align: center;">
  <img src="/images/Decoding/sampl_1.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

#### Issues with ancestral sampling.

Sampling introduces a new set of challenges because it doesn't completely eliminate the probabilities of any tokens. In vanilla sampling, every token in the vocabulary remains a potential option, which can sometimes result in generating inappropriate words. Even with a well-trained model, where most of the probability mass is concentrated on a limited set of suitable options, the distribution's tail remains long due to the extensive vocabulary. This phenomenon, known as a heavy tail distribution, is characteristic of language. Consequently, when aggregated, the probabilities of these less likely tokens can still be significant. 

For instance, many tokens may be contextually inappropriate, yet a good language model assigns each a very small probability. However, the sheer number of these tokens means that, collectively, they still have a considerable chance of being selected. To address this issue, we can cut off the tail by zeroing out the probabilities of the unwanted tokens. 

### Top $ k $ Sampling

<div style="text-align: center;">
  <img src="/images/Decoding/sampl_3.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

One effective approach is top-$k$ sampling, where we only sample from the top $k$ tokens in the probability distribution.
Increasing $k$ results in more diverse but potentially risky outputs, while decreasing $k$ leads to safer but more generic outputs.

<div style="text-align: center;">
  <img src="/images/Decoding/sampl_2.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

<div style="text-align: center;">
  <img src="/images/Decoding/sampl_4.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

Top-$k$ decoding can present two major issues. First, it can cut off too quickly, as illustrated by the sentence "She said, 'I never \___'." Many valid options, such as "won't" or "can't," might be excluded because they don't fall within the top $k$ candidates, leading to poor recall for the generation system. Second, top-$k$ decoding can also cut off too slowly. For instance, in the sentence "ate the pizza while it was still ___," the word "cold" is an unlikely choice according to common sense. Despite its low probability, the model might still sample "cold" as an output, resulting in poor precision for the generation model.

Given the problems with top-$k$ decoding, how can we address the issue that there is no single $k$ that fits all circumstances? The probability distributions we sample from are dynamic, and this variability causes issues. 
When the probability distribution is relatively flat, using a small $ k $ will eliminate many viable options, so we want $k$ to be larger in this case. Conversely, when the distribution is too peaky, a high $k$ would allow too many options to be viable. In this situation, we might want a smaller $k$ to be more selective.

### Top $P$ sampling (nucleus sampling)

The solution might be that $k$ is just a suboptimal hyperparameter. Instead of focusing on $k$, we should consider sampling from tokens within the top $P$ probability percentiles of the cumulative distribution function (CDF). This approach adapts to the shape of the probability distribution and can provide a more flexible and effective sampling method.

<div style="text-align: center;">
  <img src="/images/Decoding/sampl_5.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

The advantage of using top-$P$ sampling, where we sample from the top $P$ percentile of the cumulative probability mass, is that it effectively gives us an adaptive $k$ for each different probability distribution. 

Let me explain what I mean by adaptive $k$. In the first distribution, which follows a typical power law of language, using top-$k$ sampling would mean selecting the top $k$ tokens. However, using top-$P$ sampling means we are focusing on the top $P$ percentile of the cumulative probability, which might be similar to top-K in this case.

For a relatively flat distribution like the blue one, top-$P$ sampling includes more candidates compared to top-$k$. On the other hand, for a more skewed distribution like the green one, top-$P$ sampling includes fewer candidates.
By selecting the top $P$ percentile in the probability distribution, we achieve a more flexible $k$, which better reflects the good options in the model. This adaptive approach ensures that we are considering an appropriate number of candidates based on the shape of the distribution, leading to more effective sampling.

### Epsilon Sampling

Epsilon sampling involves setting a threshold for lower bound probabilities. Essentially, if a word's probability is less than a certain value, like 0.03, that word will never appear in the output distribution. This ensures that low-probability words are excluded from the final output.

### Temperature Scaling

Another hyperparameter that we can tune to affect decoding is the temperature parameter $ \tau $. Recall that at each time step, the model computes a score for each word, and we use the softmax function to convert these scores into a probability distribution:

$$
P_t(y_t = w) = \frac{\exp(S_w)}{\sum_{w' \in V} \exp(S_{w'})}
$$

We can adjust these scores by introducing the temperature parameter $ \tau $. Specifically, we divide all scores $ S_w $ by $ \tau $ before applying the softmax function:

$$
P_t(y_t = w) = \frac{\exp(S_w / \tau)}{\sum_{w' \in V} \exp(S_{w'} / \tau)}
$$

This temperature adjustment affects the spread of the probability distribution without changing its monotonicity. For example, if word A had a higher probability than word B before the adjustment, it will still have a higher probability afterward, though the relative differences will change.

- **Raising the temperature $( \tau > 1 )$:** The distribution $ P_t $ becomes more uniform (flatter). This means that the probabilities are more spread out across different words, leading to more diverse output.

- **Lowering the temperature $( \tau < 1 )$:** The distribution $ P_t $ becomes more spiky. Probabilities are concentrated on the top words, resulting in less diverse output. In the extreme case, if $ \tau $ approaches zero, the probability distribution becomes a one-hot vector, concentrating all the probability mass on a single word, effectively reducing the method to argmax sampling (greedy decoding).

Temperature is a hyperparameter for decoding, similar to $ k $ in top-k sampling and $ P $ in top-p sampling. It can be tuned for both beam search and sampling algorithms and is orthogonal to the approaches we discussed before. Adjusting the temperature allows us to control the diversity of the generated output, balancing between exploring a wide range of options and focusing on the most likely ones.

<div style="text-align: center;">
  <img src="/images/Decoding/temp.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

## Some other sampling methods (edit)

So all the decoding methods we discussed so far are standard decoding methods. But, just like any other area of NLP, this is an acitvely researched field. Next i am going to present some more advanced decoding methods that have popped up over the past few years that i think are relly and are being used to import the decoding enven more.
Cool and the second one is the one people suspect is used by Open AI for faster inference on their massive models like GPT4.

### Contrastive Decoding

The idea is to incorporate additional information during the decoding process of language models by utilizing another another model. If you've experimented with relatively small language models like GPT-2 small, you may have noticed that they often degenerate into repeating the same sequence or provide incorrect outputs when asked factual questions. These issues are less prevalent in larger models trained on more extensive data.

The question arises: can we use the shortcomings of the smaller model to enhance the performance of the larger model? The approach is based on the intuition that if the smaller model assigns a low probability to a certain answer while the larger model assigns a high probability, it's likely because the larger model has learned something the smaller model hasn't. Therefore, we modify the probability distribution of the larger model to favor outputs that it considers highly likely and the weaker model considers unlikely.

<div style="text-align: center;">
  <img src="/images/Decoding/contrastive.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

For example, consider the input: **"Barack Obama was born in Hawaii. He was born in..."** The smaller model might start repeating itself, and even naive sampling from the larger model can lead to repetitive loops like **"He was born in Hawaii. He was born in Hawaii..."**—a behavior we generally want to avoid. Using methods like nucleus or top-p sampling might yield factually incorrect outputs such as **"He was born in Washington D.C."**

By employing **contrastive decoding**, we take the outputs from our expert (larger) model and subtract the probabilities assigned by the weaker (smaller) model. This process emphasizes outputs that the stronger model deems probable but the weaker model does not, likely because these are facts known to the larger model but not the smaller one. In this example, we might obtain the actual year Barack Obama was born—a fact the larger model knows and the smaller model doesn't.

This method is part of a broader class of techniques that use external information to improve decoding by adjusting the probability distribution at each step.

Lets look at a few questions that i had when i read this for the first time...

**Does this approach improve upon standard methods?** Generally, yes. Both the expert and weak models might assign high probabilities to degenerate cases like repetitive sequences because they're easy patterns to learn. However, genuinely valuable outputs that only the expert model can produce tend to have low probabilities under the weak model. By subtracting the weak model's probabilities, we filter out these less desirable behaviors, retaining the high-quality outputs.

**When generating longer sequences with contrastive decoding, how do you decide when to involve the expert model?** In contrastive decoding, this adjustment occurs at every individual time step. We use the expert model to generate predictions and incorporate the amateur model to subtract probabilities for each next token. While the paper applies this at every step, you could opt to use it selectively—such as when facing high uncertainty or a less sharp probability distribution.

**How weak should the weak predictor be?** The paper doesn't suggest a significant disparity between the two models used. For instance, they experimented with GPT-2 XL and GPT-2 small, which differ in parameter counts and data but aren't drastically different in capability. The key is to choose a weak model that's not so similar to the expert that it subtracts useful information, nor so weak that it lacks any valuable insights about the task. The optimal choice may vary depending on the specific task at hand.

**Could highly plausible tokens receive low contrastive scores if both models assign them similar probabilities, potentially causing the model to overlook good continuations?** 

Yes, highly plausible tokens can receive low contrastive scores when both the expert and amateur models assign them similar probabilities, which might cause the model to overlook good continuations. To prevent this, the authors implement an **adaptive plausibility constraint** that ensures only tokens deemed sufficiently probable by the expert model are considered. By filtering out less probable tokens and having the amateur model adjust the scores of the remaining candidates, the approach maintains high-quality and distinctive continuations without discarding promising tokens.

## Speculative Decoding

In the realm of large language models, generating text efficiently without compromising quality is a significant challenge. **Speculative decoding** emerges as a powerful technique to speed up the inference process by leveraging a faster approximation model alongside the target model. This post delves into the mathematical underpinnings of speculative decoding, illustrates it with a detailed example, and discusses its advantages and practical considerations.

---

### Understanding Speculative Decoding

Speculative decoding accelerates text generation by using a lightweight approximation model to generate multiple candidate tokens in parallel, and then evaluating these candidates with the accurate but slower target model. By accepting tokens that are probable under the target model and adjusting the distribution when necessary, we ensure that the final output matches the target model's output distribution without the computational overhead of evaluating every token sequentially with the target model.

The key components of speculative decoding are:

- **Target Model ($ M_p $)**: The accurate language model providing the true probability distribution $ p(x_t | x_{<t}) $.
- **Approximation Model ($ M_q $)**: A faster, less accurate model approximating $ M_p $, providing $ q(x_t | x_{<t}) $.
- **Prefix ($ x_{<t} $)**: The sequence of tokens generated so far.
- **Completion Parameter ($ \gamma $)**: The number of speculative tokens generated.

---

## The Speculative Decoding Algorithm

The speculative decoding process combines outputs from both the approximation model $ M_q $ and the target model $ M_p $ in a single decoding step.

**Step 1: Sampling Guesses from the Approximation Model**

We begin by using the approximation model $ M_q $ to generate $ \gamma $ speculative tokens. For each speculative token, we compute the probability distribution $ q_i(x) $ based on the current prefix and sample a token $ x_i $ from this distribution. This process is autoregressive, meaning each token is generated based on the prefix plus any previously generated speculative tokens.

**Step 2: Evaluating with the Target Model**

In parallel, we use the target model $ M_p $ to compute the true probability distributions $ p_i(x) $ for each extended prefix resulting from the tokens sampled by $ M_q $. This allows us to assess the likelihood of the speculative tokens under the target model.

**Step 3: Determining Acceptance of Speculative Tokens**

For each speculative token $ x_i $, we determine whether to accept it based on an acceptance criterion. We calculate the acceptance probability $ a_i = \min\left(1, \frac{p_i(x_i)}{q_i(x_i)}\right) $ and generate a random number $ r_i $ from a uniform distribution between 0 and 1. If $ r_i \leq a_i $, we accept the token. We continue this process sequentially; if a token is rejected, we stop accepting further speculative tokens because their contexts include the rejected token.

**Step 4: Adjusting the Distribution if Needed**

If we reject a token or reach the end of the speculative tokens, we adjust the probability distribution for sampling the next token from the target model. We compute an adjusted distribution $ p'(x) = \text{normalize}(\max(0, p_{n+1}(x) - q_{n+1}(x))) $, where $ n $ is the number of accepted speculative tokens. This adjustment ensures that we account for the probability mass not covered by the accepted speculative tokens.

**Step 5: Generating the Next Token**

Finally, we sample the next token $ t $ from the adjusted distribution $ p'(x) $ using the target model $ M_p $. We then construct the new prefix by concatenating the accepted speculative tokens $ x_1, ..., x_n $ and the new token $ t $.

---

## Detailed Example

To illustrate the speculative decoding algorithm, let's walk through a concrete example.

**Assumptions:**

- **Target Model ($ M_p $)**: An accurate but slower language model.
- **Approximation Model ($ M_q $)**: A faster, less accurate model.
- **Prefix**: "The quick brown".
- **Vocabulary**: Includes words like "fox", "dog", "cat", "jumps", etc.
- **Completion Parameter ($ \gamma $)**: Set to 2.

**Step 1: Sampling Guesses from $ M_q $**

*Iteration 1 ($ i = 1 $)*

We run $ M_q $ on the prefix "The quick brown" to compute $ q_1(x) $. Suppose $ q_1(\text{fox}) = 0.6 $, $ q_1(\text{dog}) = 0.3 $, and $ q_1(\text{cat}) = 0.1 $. We sample $ x_1 $ from $ q_1(x) $, and let's say we get $ x_1 = \text{fox} $.

*Iteration 2 ($ i = 2 $)*

Next, we run $ M_q $ on the extended prefix "The quick brown fox" to compute $ q_2(x) $. Suppose $ q_2(\text{jumps}) = 0.5 $, $ q_2(\text{sleeps}) = 0.3 $, and $ q_2(\text{runs}) = 0.2 $. We sample $ x_2 $ from $ q_2(x) $, and let's say we get $ x_2 = \text{jumps} $.

**Step 2: Evaluating with $ M_p $**

We compute the true probability distributions using $ M_p $:

- $ p_1(x) $ for the prefix "The quick brown": Suppose $ p_1(\text{fox}) = 0.7 $, $ p_1(\text{dog}) = 0.2 $, $ p_1(\text{cat}) = 0.1 $.
- $ p_2(x) $ for the prefix "The quick brown fox": Suppose $ p_2(\text{jumps}) = 0.6 $, $ p_2(\text{sleeps}) = 0.2 $, $ p_2(\text{runs}) = 0.2 $.
- $ p_3(x) $ for the prefix "The quick brown fox jumps": Suppose $ p_3(\text{over}) = 0.8 $, $ p_3(\text{high}) = 0.1 $, $ p_3(\text{quickly}) = 0.1 $.

**Step 3: Determining Acceptance**

*For $ x_1 = \text{fox} $:*

We calculate the acceptance probability $ a_1 = \min\left(1, \frac{p_1(\text{fox})}{q_1(\text{fox})}\right) = \min\left(1, \frac{0.7}{0.6}\right) = 1 $. We generate a random number $ r_1 $, say $ r_1 = 0.5 $. Since $ r_1 \leq a_1 $, we accept $ x_1 = \text{fox} $.

*For $ x_2 = \text{jumps} $:*

We calculate $ a_2 = \min\left(1, \frac{p_2(\text{jumps})}{q_2(\text{jumps})}\right) = \min\left(1, \frac{0.6}{0.5}\right) = 1 $. We generate $ r_2 = 0.8 $. Since $ r_2 \leq a_2 $, we accept $ x_2 = \text{jumps} $.

We have accepted both speculative tokens, so $ n = 2 $.

**Step 4: Adjusting the Distribution**

Since we have accepted all speculative tokens, we adjust the distribution for the next token. We set $ p'(x) = p_3(x) $ and compute $ q_3(x) $ using $ M_q $ on the prefix "The quick brown fox jumps". Suppose $ q_3(\text{over}) = 0.7 $, $ q_3(\text{high}) = 0.2 $, and $ q_3(\text{quickly}) = 0.1 $.

We adjust $ p'(x) $ by subtracting $ q_3(x) $ and normalizing:

- $ p'(\text{over}) = \max(0, 0.8 - 0.7) = 0.1 $
- $ p'(\text{high}) = \max(0, 0.1 - 0.2) = 0 $
- $ p'(\text{quickly}) = \max(0, 0.1 - 0.1) = 0 $

After normalization (since total mass is 0.1), we have $ p'(\text{over}) = 1.0 $.

**Step 5: Generating the Next Token**

We sample $ t $ from $ p'(x) $, which must be "over" since $ p'(\text{over}) = 1.0 $. We construct the new prefix by adding the accepted tokens and the new token: "The quick brown **fox jumps over**".

---

## Theoretical Justification

Speculative decoding uses an acceptance-rejection sampling mechanism to correct for discrepancies between the approximation model $ M_q $ and the target model $ M_p $. By accepting samples from $ M_q $ with a probability proportional to $ \frac{p(x)}{q(x)} $, we ensure that the accepted tokens follow the target distribution $ p(x) $.

When tokens are rejected, we adjust the distribution for the next token to account for the probability mass not covered by $ M_q $. This adjustment ensures that the final output aligns with the target model's distribution, preserving the quality of the generated text.

---

## Practical Considerations

**Choosing the Completion Parameter ($ \gamma $)**

There is a trade-off in selecting $ \gamma $. A larger $ \gamma $ increases the chance of accepting more tokens, potentially accelerating the decoding process, but it also adds computational overhead due to the increased number of computations required by $ M_p $ and $ M_q $. It's important to balance $ \gamma $ based on model performance and available resources.

**Quality of the Approximation Model**

The effectiveness of speculative decoding depends on the approximation quality of $ M_q $. A closer match between $ q(x) $ and $ p(x) $ leads to higher acceptance rates, improving efficiency. However, $ M_q $ must be significantly faster than $ M_p $ to justify its use.

**Computational Resources**

Speculative decoding requires hardware capable of parallel processing to run $ M_p $ on multiple prefixes simultaneously. This may increase memory usage and computational demands, which should be considered when implementing this technique.

**Implementation Tips**

- Utilize batch processing to optimize computations.
- Cache repeated computations when possible to reduce redundancy.
- Handle calculations carefully to prevent numerical instability, such as division by zero or overflow errors.

---

## Advantages and Limitations

**Advantages**

Speculative decoding reduces the number of sequential steps required by $ M_p $, accelerating the inference process. The technique is flexible, compatible with various sampling methods, and can be adapted to different models. Importantly, the final outputs align with the target model's distribution, maintaining text generation quality.

**Limitations**

The algorithm adds complexity to the decoding process, requiring careful implementation. There may be increased computational resource requirements due to parallel computations. Additionally, the effectiveness of speculative decoding relies on the approximation quality of $ M_q $; if $ M_q $ is not a good approximation of $ M_p $, the acceptance rate may be low, reducing the efficiency gains.

---

## Conclusion

Speculative decoding offers an innovative approach to speeding up language model inference without sacrificing output quality. By combining a fast approximation model with an acceptance-rejection mechanism, we can generate multiple tokens in parallel, significantly accelerating the decoding process. This technique is particularly beneficial when working with large models where inference speed is a bottleneck.

---

*Note: This post simplifies complex mathematical concepts for clarity. For a deeper mathematical understanding, refer to the original research papers on speculative decoding and sampling methods in language models.*

---

This detailed explanation covers the theoretical foundations, step-by-step procedures, practical considerations, and potential benefits and limitations of speculative decoding, providing a comprehensive understanding of the algorithm.