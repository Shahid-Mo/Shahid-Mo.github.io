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

   At the initial step $ t = 0 $, we start with the beam containing just the start token $\langle s \rangle$:

   $$
   \text{Beam}_0 = \{(\langle s \rangle, 0)\}
   $$

2. **Expanding Sequences:**

   For each sequence $ Y_{1:t-1} $ in the beam, we generate all possible extensions by appending each possible next token $ \hat{y_t} $. Each extension forms a new candidate sequence:

   $$
   Y_{1:t} = Y_{1:t-1} \oplus \hat{y_t}
   $$

3. **Computing Log Probabilities:**

   For each new candidate sequence $ Y_{1:t} $, we compute the total log probability:

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

    Select the highest-scoring complete sequence from the final beam.

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

Another hyperparameter that we can tune to affect decoding is the temperature parameter $$ \tau $$. Recall that at each time step, the model computes a score for each word, and we use the softmax function to convert these scores into a probability distribution:

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

## Some other sampling methods

So all the decoding methods we discussed so far are standard decoding methods, just like any other area of NLP, this is an acitvely researched field. Next i am going to present some more advanced decoding methods that have popped up over the past few years that i think are relly and are being used to import the decoding enven more.
Cool and the second one is the one people suspect is used by Open AI for faster inference on their massive models like GPT4.

### Contrastive Decoding

The idea is to incorporate additional information during the decoding process of language models by utilizing another another model. If you've experimented with relatively small language models like GPT-2 small, you may have noticed that they often degenerate into repeating the same sequence or provide incorrect outputs when asked factual questions. These issues are less prevalent in larger models trained on more extensive data.

The question arises: can we use the shortcomings of the smaller model to enhance the performance of the larger model? The approach is based on the intuition that if the smaller model assigns a low probability to a certain answer while the larger model assigns a high probability, it's likely because the larger model has learned something the smaller model hasn't. Therefore, we modify the probability distribution from the larger model to favor outputs that it considers highly likely and the weaker model considers unlikely.

<div style="text-align: center;">
  <img src="/images/Decoding/contrastive.png" alt="TF32 Explained" style="display: block; margin: 0 auto;">
<p style="font-size: 0.8em; color: rgba(0, 0, 0, 0.6);">
  Figure 1: Comparison of FP8 and BF16 formats. Source: 
  <a href="https://arxiv.org/abs/xxxx.xxxxx" style="color: rgba(0, 0, 0, 0.6);">Smith et al. (2023)</a>
</p>
</div>

For example, consider the input: **"Barack Obama was born in Hawaii. He was born in..."** The smaller model might start repeating itself, and even naive sampling from the larger model can lead to repetitive loops like **"He was born in Hawaii. He was born in Hawaii..."**—a behavior we generally want to avoid. Using methods like nucleus or top-p sampling might yield factually incorrect outputs such as **"He was born in Washington D.C."**

By employing **contrastive decoding**, we take the outputs from our expert (larger) model and subtract the probabilities assigned by the weaker (smaller) model. This process emphasizes outputs that the stronger model deems probable but the weaker model does not, likely because these are facts known to the larger model but not the smaller one. In this example, we might obtain the actual year Barack Obama was born—a fact the larger model knows and the smaller model doesn't.

This method is part of a broader class of techniques that use external information to improve decoding by adjusting the probability distribution at each step. These techniques offer alternative sampling strategies before delving into search-based methods.

**Does this approach improve upon standard methods?** Generally, yes. Both the expert and weak models might assign high probabilities to degenerate cases like repetitive sequences because they're easy patterns to learn. However, genuinely valuable outputs that only the expert model can produce tend to have low probabilities under the weak model. By subtracting the weak model's probabilities, we filter out these less desirable behaviors, retaining the high-quality outputs.

**When generating longer sequences with contrastive decoding, how do you decide when to involve the expert model?** In contrastive decoding, this adjustment occurs at every individual time step. We use the expert model to generate predictions and incorporate the amateur model to subtract probabilities for each next token. While the paper applies this at every step, you could opt to use it selectively—such as when facing high uncertainty or a less sharp probability distribution.

**How weak should the weak predictor be?** The paper doesn't suggest a significant disparity between the two models used. For instance, they experimented with GPT-2 XL and GPT-2 small, which differ in parameter counts and data but aren't drastically different in capability. The key is to choose a weak model that's not so similar to the expert that it subtracts useful information, nor so weak that it lacks any valuable insights about the task. The optimal choice may vary depending on the specific task at hand.

**Is this method applicable during inference?** Yes, this technique is used during inference and doesn't require retraining the models. Everything discussed can be directly applied during the decoding process to enhance output quality.

**Contrastive Decoding: Mathematical Details Explained**

**1. Problem Statement**

In open-ended language generation, we aim to generate fluent and coherent text continuations given a prompt. Formally, we have:

- **Prompt (Prefix)**: A sequence of tokens $ x_{pre} = x_1, x_2, \dots, x_n $.
- **Continuation**: A sequence of tokens $ x_{cont} = x_{n+1}, x_{n+2}, \dots, x_{n+m} $ that we want to generate.

We use a pre-trained autoregressive language model (LM) to generate the continuation by predicting one token at a time, conditioned on all previous tokens:

$$
p_{LM}(x_{cont} \mid x_{pre}) = \prod_{i=n+1}^{n+m} p_{LM}(x_i \mid x_{<i})
$$

where $ x_{<i} = x_1, x_2, \dots, x_{i-1} $ represents the context up to token $ x_i $.

We introduce two types of language models:

- **Expert LM ($ p_{EXPERT} $)**: A larger, more capable model (e.g., GPT-2 XL).
- **Amateur LM ($ p_{AMA} $)**: A smaller, less capable model (e.g., GPT-2 small).

**2. Contrastive Decoding Objective**

The core idea of contrastive decoding is to leverage the strengths of the expert LM while mitigating common errors by comparing its predictions with those of the amateur LM. We define the **contrastive objective** as:

$$
L_{CD}(x_{cont}; x_{pre}) = \log p_{EXPERT}(x_{cont} \mid x_{pre}) - \log p_{AMA}(x_{cont} \mid x_{pre})
$$

This objective rewards continuations that the expert LM deems likely but the amateur LM does not. By subtracting the amateur's log-probabilities, we penalize patterns that are common mistakes of smaller models, such as repetitions or incoherence.

**3. Challenges with the Contrastive Objective**

While the contrastive objective helps highlight the expert LM's strengths, it can introduce two issues:

- **False Positives**: Implausible tokens may receive high contrastive scores if the amateur LM assigns them very low probabilities, even if the expert LM also considers them unlikely.
- **False Negatives**: Highly plausible tokens may receive low contrastive scores if both models assign them similar probabilities, potentially causing the model to overlook good continuations.

**4. Adaptive Plausibility Constraint**

To address these challenges, we introduce an **adaptive plausibility constraint**. This constraint ensures that we only consider tokens that the expert LM deems sufficiently probable.

**Definition**:

$$
\hat{head}(x_i) = \lbrace { x \in V : p\_{EXPERT}(x | x\_{<i}) \geq \alpha \cdot \max\_{x' \in V} p\_{EXPERT}(x' | x\_{<i}) } \rbrace
$$

- **$ V $**: The vocabulary of possible tokens.
- **$ \alpha \in [0, 1] $**: A hyperparameter that controls the cutoff threshold.
  - **$ \alpha $ close to 1**: Only the most probable tokens are included.
  - **$ \alpha $ close to 0**: A wider range of tokens is included.

**Purpose**:

- **Prevent False Positives**: Excludes tokens that the expert LM considers implausible, even if they have high contrastive scores.
- **Prevent False Negatives**: Ensures that highly probable tokens (according to the expert LM) are not discarded due to low contrastive scores.

**5. Full Contrastive Decoding Method**

Combining the contrastive objective with the adaptive plausibility constraint, we define the **token-level contrastive score** for each candidate token $ x_i $:

$$
CD\_score(x_i; x_{<i}) = 
\begin{cases} 
\log \dfrac{p_{EXPERT}(x_i \mid x_{<i})}{p_{AMA}(x_i \mid x_{<i})}, & \text{if } x_i \in \hat{head}(x_i) \\
-\infty, & \text{otherwise}
\end{cases}
$$

- **Interpretation**:
  - If a token $ x_i $ is within the plausible set $ \hat{head}(x_i) $, we compute its contrastive score.
  - If not, the token is assigned a score of $ -\infty $, effectively eliminating it from consideration.

**Sequence-Level Objective**:

The overall objective for the continuation $ x_{cont} $ becomes:

$$
L_{CD}(x_{cont}; x_{pre}) = \sum_{i=n+1}^{n+m} CD\_score(x_i; x_{<i})
$$

**Decoding Process**:

1. **At Each Time Step $ i $**:
   - **Compute $ p_{EXPERT}(x \mid x_{<i}) $**: Obtain the expert LM's next-token probabilities.
   - **Apply Plausibility Constraint**:
     - Form $ \hat{head}(x_i) $ by selecting tokens satisfying the constraint.
   - **Compute $ CD\_score(x_i; x_{<i}) $** for tokens in $ \hat{head}(x_i) $.

2. **Beam Search**:
   - Use beam search to explore possible continuations.
   - At each step, expand beams by selecting tokens with the highest $ CD\_score $.
   - Discard beams that end with tokens not in $ \hat{head}(x_i) $.

**6. Practical Considerations**

**Choice of $ \alpha $ in Plausibility Constraint**:

- A balance is necessary:
  - **High $ \alpha $**: May be too restrictive, potentially missing good continuations.
  - **Low $ \alpha $**: May allow less plausible tokens, reintroducing false positives.
- Typically, $ \alpha $ is set to a value like 0.1, allowing for a reasonable diversity of plausible tokens.

**Beam Width**:

- Determines the number of candidate continuations explored.
- Larger beam widths increase computational cost but can improve the quality of the generated text.

**7. Selection of the Amateur LM**

The amateur LM should be chosen carefully to maximize the effectiveness of contrastive decoding.

**Factors to Consider**:

- **Model Size**:
  - A smaller model within the same family as the expert LM.
  - Example: If the expert is GPT-2 XL, the amateur might be GPT-2 small.
- **Temperature Adjustment**:
  - Adjust the **temperature parameter $ \tau $** of the amateur LM to control the sharpness of its probability distribution.
    - **High $ \tau $** (>1): Flattens the distribution, making the amateur less confident.
    - **Low $ \tau $** (<1): Sharpens the distribution, emphasizing the amateur's mistakes.
- **Context Window**:
  - Limit the amateur LM's context to a smaller window (e.g., only the last few tokens), reducing its ability to maintain long-term coherence.
  
**8. Example Walkthrough**

Let's illustrate how contrastive decoding works with an example:

- **Given Prompt**: "Barack Obama was born in Hawaii. He was born in"
- **Expert LM Predictions**:
  - High probabilities for tokens like "1961", "August", "Honolulu".
- **Amateur LM Predictions**:
  - May assign high probabilities to repetitive patterns like "Hawaii", "the".

**Steps**:

1. **Compute $ p_{EXPERT}(x_i \mid x_{<i}) $** and $ p_{AMA}(x_i \mid x_{<i}) $** for all tokens.
2. **Apply Plausibility Constraint**:
   - Identify tokens where $ p_{EXPERT}(x_i \mid x_{<i}) $ is within $ \alpha $ of the maximum probability.
3. **Calculate $ CD\_score(x_i; x_{<i}) $**:
   - For plausible tokens, compute the contrastive score.
   - Tokens like "1961" may have high contrastive scores because they are plausible under the expert LM and not favored by the amateur LM.
4. **Select Next Token**:
   - Choose the token with the highest $ CD\_score $.
   - This leads to selecting informative and coherent continuations.

**9. Advantages of Contrastive Decoding**

- **Reduces Repetition**: Penalizes repetitive patterns commonly produced by smaller models.
- **Enhances Coherence**: Highlights the expert LM's ability to maintain context and generate coherent text.
- **Balances Creativity and Plausibility**: Generates text that is both plausible (as judged by the expert LM) and original (differing from the amateur LM's tendencies).

**10. Summary**

Contrastive decoding effectively combines the strengths of a large language model with the corrective influence of a smaller model to generate high-quality text. By mathematically defining a contrastive objective and applying an adaptive plausibility constraint, we can navigate the trade-offs between plausibility and originality, resulting in more coherent and fluent language generation.

**Key Equations**:

- **Contrastive Objective**:

  $$
  L_{CD}(x_{cont}; x_{pre}) = \log p_{EXPERT}(x_{cont} \mid x_{pre}) - \log p_{AMA}(x_{cont} \mid x_{pre})
  $$

- **Adaptive Plausibility Constraint**:

$$
\hat{head}(x_i) = \lbrace { x \in V : p\_{EXPERT}(x | x\_{<i}) \geq \alpha \cdot \max\_{x' \in V} p\_{EXPERT}(x' | x\_{<i}) } \rbrace
$$

- **Token-Level Contrastive Score**:

  $$
  CD\_score(x_i; x_{<i}) = 
  \begin{cases} 
  \log \dfrac{p_{EXPERT}(x_i \mid x_{<i})}{p_{AMA}(x_i \mid x_{<i})}, & \text{if } x_i \in \hat{head}(x_i) \\
  -\infty, & \text{otherwise}
  \end{cases}
  $$

By understanding and applying these mathematical formulations, we can implement contrastive decoding to enhance language generation tasks effectively.

# Speculative Decoding

This is the decoding stratergy that people suspect that Open AI uses for its GPT class of Models.

# Speculative Decoding: A Detailed Explanation

## 2. Speculative Decoding

Speculative decoding is a technique designed to accelerate the inference process of large language models by leveraging a more efficient approximation model alongside the target model. This method allows us to generate multiple tokens in parallel, potentially reducing the computation time while maintaining the quality of the generated text.

### 2.1 Overview

**Definitions:**

- **Target Model ($ M_p $)**: The primary language model we aim to accelerate. It provides the true probability distribution $ p(x_t | x_{<t}) $ for the next token $ x_t $ given the previous tokens $ x_{<t} $.
- **Approximation Model ($ M_q $)**: A more efficient but less accurate model that approximates $ M_p $. It provides the distribution $ q(x_t | x_{<t}) $.
- **Prefix ($ x_{<t} $)**: The sequence of tokens generated so far, serving as the context for predicting the next token.
- **Completion Parameter ($ \gamma $)**: A positive integer indicating the number of speculative completions generated by $ M_q $.

**Core Idea:**

1. **Speculative Generation with $ M_q $:**
   - Use $ M_q $ to generate $ \gamma $ possible continuations (tokens) from the current prefix. This is done autoregressively, meaning each subsequent token is generated based on the prefix plus the previously generated tokens.

2. **Parallel Evaluation with $ M_p $:**
   - Run $ M_p $ to compute the true probability distributions for the generated tokens from $ M_q $. This is done in parallel to save time.

3. **Acceptance and Adjustment:**
   - **Acceptance Criterion:** If a token generated by $ M_q $ has a probability under $ M_p $ that is at least as high as under $ M_q $, it's accepted.
   - **Rejection and Resampling:** If not, the token is rejected with a certain probability, and a new token is sampled from an adjusted distribution derived from $ M_p $.

**Benefits:**

- **Efficiency:** By potentially accepting multiple tokens at once, we reduce the number of sequential computations required by $ M_p $.
- **Guaranteed Quality:** The method ensures that the final output distribution matches that of $ M_p $, maintaining the quality of the generated text.

### 2.2 Standardized Sampling

**Sampling Methods:**

- **Argmax Sampling:** Select the token with the highest probability.
- **Top-k Sampling:** Consider only the top $ k $ tokens with the highest probabilities.
- **Nucleus (Top-p) Sampling:** Consider the smallest set of tokens whose cumulative probability exceeds a threshold $ p $.
- **Temperature Scaling:** Adjust the probabilities by scaling logits before applying softmax.

**Standardization:**

All these sampling methods can be unified under the framework of sampling from an adjusted probability distribution. This standardization allows us to:

- Treat different sampling methods uniformly.
- Apply the speculative decoding technique regardless of the sampling strategy used.
- Simplify the theoretical analysis and implementation.

### 2.3 Speculative Sampling

**Objective:** To sample a token $ x $ from the target distribution $ p(x) $ using the approximation distribution $ q(x) $.

**Procedure:**

1. **Initial Sampling:**
   - Sample $ x \sim q(x) $.

2. **Acceptance Criterion:**
   - **If $ q(x) \leq p(x) $:** Accept $ x $ with certainty.
   - **If $ q(x) > p(x) $:** Accept $ x $ with probability $ \frac{p(x)}{q(x)} $.

3. **Rejection and Resampling:**
   - If $ x $ is rejected, sample a new token from the adjusted distribution:
     $$
     p'(x) = \text{normalize}(\max(0, p(x) - q(x)))
     $$
   - This adjustment ensures that the overall sampling process is unbiased and that $ x $ is ultimately drawn from $ p(x) $.

**Theoretical Justification:**

- The acceptance-rejection mechanism preserves the target distribution $ p(x) $.
- The adjusted distribution $ p'(x) $ accounts for the probability mass not covered by $ q(x) $.

**Algorithm Overview:**

- **Generate Guesses:** Use $ M_q $ to produce $ \gamma $ candidate tokens.
- **Evaluate with $ M_p $:** Compute the probabilities of these tokens under $ M_p $.
- **Acceptance Check:** Use an acceptance-rejection test for each token.
- **Adjustment:** If a token is rejected, adjust the distribution accordingly and sample from $ M_p $.

## Algorithm 1: SpeculativeDecodingStep

**Purpose:** To generate one or more tokens in a single decoding step, combining outputs from $ M_q $ and $ M_p $.

**Inputs:**

- **$ M_p $:** Target model.
- **$ M_q $:** Approximation model.
- **$ \text{prefix} $:** Current sequence of tokens.

**Steps:**

### Step 1: Sample $ \gamma $ Guesses from $ M_q $

**Process:**

For $ i = 1 $ to $ \gamma $:

1. **Compute $ q_i(x) $:**
   - Run $ M_q $ on $ \text{prefix} + [x_1, ..., x_{i-1}] $ to get the probability distribution for the next token.
2. **Sample $ x_i $ from $ q_i(x) $:**
   - Use the computed distribution to sample the next token $ x_i $.

**Explanation:**

- **Autoregressive Generation:** Each token $ x_i $ is generated based on all previous tokens, including those just generated.
- **Parallelization Potential:** This step can be efficiently computed due to $ M_q $ being more lightweight than $ M_p $.

### Step 2: Run $ M_p $ in Parallel

**Process:**

- **Compute Distributions:**
  - Run $ M_p $ on $ \text{prefix} $ and each extended prefix to get $ p_1(x), ..., p_{\gamma + 1}(x) $.
    - $ p_1(x) $: Distribution after $ \text{prefix} $.
    - $ p_2(x) $: Distribution after $ \text{prefix} + [x_1] $.
    - Continue up to $ p_{\gamma + 1}(x) $ after all $ \gamma $ tokens.

**Explanation:**

- **Parallel Computation:** Since the prefixes differ only by the added tokens from $ M_q $, $ M_p $ can process them simultaneously.
- **Preparation for Acceptance Check:** These distributions are needed to determine whether to accept the tokens sampled from $ M_q $.

### Step 3: Determine the Number of Accepted Guesses

**Process:**

1. **Generate Random Numbers:**
   - For each $ i $, sample $ r_i \sim U(0,1) $, where $ U(0,1) $ denotes the uniform distribution between 0 and 1.

2. **Acceptance Check:**
   - For each $ i $ from 1 to $ \gamma $, check if:
     $$
     r_i \leq \frac{p_i(x_i)}{q_i(x_i)}
     $$
   - If the condition is met, $ x_i $ is accepted.

3. **Determine $ n $:**
   - $ n $ is the number of tokens accepted.
   - If a token fails the acceptance check, we stop accepting further tokens.
   - If all tokens pass, $ n = \gamma $.

**Explanation:**

- **Acceptance Probability:** The ratio $ \frac{p_i(x_i)}{q_i(x_i)} $ represents the likelihood that $ x_i $ is a good sample under $ M_p $ given it was sampled from $ q_i(x) $.
- **Sequential Acceptance:** Once a token is rejected, subsequent tokens are not considered because their contexts include the rejected token.

### Step 4: Adjust the Distribution from $ M_p $ if Needed

**Process:**

1. **Set $ p'(x) $:**
   - $ p'(x) = p_{n+1}(x) $, the distribution from $ M_p $ after the last accepted token.

2. **Adjust the Distribution:**
   - Compute:
     $$
     p'(x) = \text{normalize}(\max(0, p'(x) - q_{n+1}(x)))
     $$
   - This step subtracts the influence of $ q_{n+1}(x) $ to adjust for tokens not accepted.

**Explanation:**

- **Purpose of Adjustment:** Ensures that the probability mass attributed to the rejected tokens in $ q_{n+1}(x) $ doesn't bias the sampling from $ p'(x) $.
- **Normalization:** Necessary to make $ p'(x) $ a valid probability distribution after the subtraction.

### Step 5: Return One Token from $ M_p $ and $ n $ Tokens from $ M_q $

**Process:**

1. **Sample Next Token $ t $:**
   - Sample $ t $ from the adjusted distribution $ p'(x) $.

2. **Construct New Prefix:**
   - Concatenate the accepted tokens $ x_1, ..., x_n $ and the new token $ t $ to the prefix.
   - Return:
     $$
     \text{prefix} + [x_1, ..., x_n, t]
     $$

**Explanation:**

- **Final Output:** The sequence includes tokens from $ M_q $ (if accepted) and a token from $ M_p $.
- **Guarantee of Correctness:** Despite using $ M_q $, the acceptance-rejection mechanism ensures the overall distribution matches that of $ M_p $.

## Detailed Example

Let's walk through the algorithm step by step with a concrete example.

**Assumptions:**

- **$ M_p $:** A large, accurate language model.
- **$ M_q $:** A smaller, faster approximation of $ M_p $.
- **Prefix:** The current text is "The quick brown".
- **Vocabulary:** Contains words like "fox", "dog", "cat", etc.
- **Completion Parameter ($ \gamma $)**: Set to 2.

### Step 1: Sample Guesses from $ M_q $

**Iteration 1 ($ i = 1 $):**

- **Compute $ q_1(x) $:**
  - Run $ M_q $ on "The quick brown".
  - Suppose $ q_1(fox) = 0.6 $, $ q_1(dog) = 0.3 $, $ q_1(cat) = 0.1 $.

- **Sample $ x_1 $:**
  - Let's say we sample $ x_1 = fox $.

**Iteration 2 ($ i = 2 $):**

- **Compute $ q_2(x) $:**
  - Run $ M_q $ on "The quick brown fox".
  - Suppose $ q_2(jumps) = 0.5 $, $ q_2(sleeps) = 0.3 $, $ q_2(runs) = 0.2 $.

- **Sample $ x_2 $:**
  - Let's say we sample $ x_2 = jumps $.

### Step 2: Run $ M_p $ in Parallel

**Compute $ p_1(x) $:**

- Run $ M_p $ on "The quick brown".
- Suppose $ p_1(fox) = 0.7 $, $ p_1(dog) = 0.2 $, $ p_1(cat) = 0.1 $.

**Compute $ p_2(x) $:**

- Run $ M_p $ on "The quick brown fox".
- Suppose $ p_2(jumps) = 0.6 $, $ p_2(sleeps) = 0.2 $, $ p_2(runs) = 0.2 $.

**Compute $ p_3(x) $:**

- Run $ M_p $ on "The quick brown fox jumps".
- Suppose $ p_3(over) = 0.8 $, $ p_3(high) = 0.1 $, $ p_3(quickly) = 0.1 $.

### Step 3: Determine Acceptance

**For $ x_1 = fox $:**

- **Calculate Acceptance Probability:**
  - $ a_1 = \frac{p_1(fox)}{q_1(fox)} = \frac{0.7}{0.6} \approx 1.17 $.
  - Since $ a_1 > 1 $, set $ a_1 = 1 $ (probability cannot exceed 1).

- **Sample $ r_1 \sim U(0,1) $:**
  - Let's say $ r_1 = 0.5 $.

- **Acceptance Check:**
  - $ r_1 \leq a_1 $ (0.5 ≤ 1), so accept $ x_1 = fox $.

**For $ x_2 = jumps $:**

- **Calculate Acceptance Probability:**
  - $ a_2 = \frac{p_2(jumps)}{q_2(jumps)} = \frac{0.6}{0.5} = 1.2 $.
  - Again, set $ a_2 = 1 $.

- **Sample $ r_2 \sim U(0,1) $:**
  - Let's say $ r_2 = 0.8 $.

- **Acceptance Check:**
  - $ r_2 \leq a_2 $ (0.8 ≤ 1), so accept $ x_2 = jumps $.

**Determine $ n $:**

- Both tokens are accepted, so $ n = 2 $.

### Step 4: Adjust the Distribution from $ M_p $

**Set $ p'(x) = p_3(x) $:**

- $ p_3(x) $ is the distribution after "The quick brown fox jumps".

**Adjust $ p'(x) $:**

- Compute $ q_3(x) $ from $ M_q $ on "The quick brown fox jumps":
  - Suppose $ q_3(over) = 0.7 $, $ q_3(high) = 0.2 $, $ q_3(quickly) = 0.1 $.

- Adjust $ p'(x) = \text{normalize}(\max(0, p_3(x) - q_3(x))) $:
  - Subtract $ q_3(x) $ from $ p_3(x) $:
    - $ p'(over) = 0.8 - 0.7 = 0.1 $
    - $ p'(high) = 0.1 - 0.2 = 0 $ (set to 0 since negative)
    - $ p'(quickly) = 0.1 - 0.1 = 0 $
  - Normalize $ p' $:
    - Total probability mass: 0.1
    - $ p'(over) = 1.0 $

### Step 5: Return Tokens

**Sample Next Token $ t $:**

- Sample $ t $ from $ p'(x) $:
  - Since $ p'(over) = 1.0 $, $ t = over $.

**Construct New Prefix:**

- Return:
  - $ \text{prefix} + [fox, jumps, over] $
  - "The quick brown fox jumps over"

**Final Output:**

- The algorithm successfully generated three tokens in one step, accelerating the decoding process while ensuring that the output aligns with $ M_p $.

## Theoretical Underpinnings

### Acceptance-Rejection Sampling

- **Purpose:** To ensure that samples drawn from $ q(x) $ match the target distribution $ p(x) $.
- **Mechanism:** By accepting samples with probability $ \frac{p(x)}{q(x)} $, we correct for any discrepancies between $ q(x) $ and $ p(x) $.
- **Guarantee:** Over many samples, the distribution of accepted tokens converges to $ p(x) $.

### Adjusted Distribution $ p'(x) $

- **Why Subtract $ q(x) $:** To remove the probability mass of tokens already considered (and possibly rejected) from $ q(x) ).
- **Normalization:** Ensures the adjusted distribution is valid (sums to 1).
- **Result:** The adjusted distribution represents the remaining "uncounted" probability mass in $ p(x) $.

## Practical Considerations

### Choice of $ \gamma $

- **Trade-off:** A larger $ \gamma $ increases the chance of accepting more tokens but also increases computational overhead.
- **Optimal $ \gamma $:** Depends on the balance between the speed of $ M_q $ and the acceptance rate.
- **Guidelines:** Choose $ \gamma $ based on empirical performance and resource constraints.

### Quality of $ M_q $

- **Approximation Accuracy:** The closer $ q(x) $ is to $ p(x) $, the higher the acceptance rate.
- **Efficiency vs. Accuracy:** $ M_q $ should be significantly faster than $ M_p $ while providing a reasonable approximation.

### Computational Resources

- **Parallel Processing:** Requires sufficient computational resources to run $ M_p $ on multiple prefixes simultaneously.
- **Memory Usage:** Increased memory consumption due to multiple instances of $ M_p $ and $ M_q $.

### Implementation Tips

- **Batch Processing:** Utilize batch computations to process multiple tokens efficiently.
- **Caching:** Cache computations where possible to avoid redundant calculations.
- **Numerical Stability:** Ensure calculations of $ \frac{p(x)}{q(x)} $ are numerically stable to avoid division by zero or overflow.

## Advantages and Limitations

### Advantages

- **Speed:** Potentially reduces the number of sequential steps required by $ M_p $, accelerating the generation process.
- **Flexibility:** Compatible with various sampling methods and can be adapted to different models.
- **Quality Preservation:** Maintains the quality of outputs as they are ultimately drawn from $ M_p $.

### Limitations

- **Complexity:** Adds complexity to the decoding process, requiring careful implementation.
- **Resource Intensive:** May require more computational resources due to parallel computations.
- **Dependent on $ M_q $:** Effectiveness is tied to the quality of the approximation model.

## Conclusion

Speculative decoding offers a powerful method to accelerate language model inference by intelligently combining a faster approximation model with the target model. By carefully accepting or rejecting tokens from the approximation model based on the target model's evaluations, we can generate multiple tokens in parallel without compromising the integrity of the final output distribution. This approach is particularly valuable when working with large models where inference speed is a critical concern.

---

This detailed explanation covers the theoretical foundations, step-by-step procedures, practical considerations, and potential benefits and limitations of speculative decoding, providing a comprehensive understanding of the algorithm.