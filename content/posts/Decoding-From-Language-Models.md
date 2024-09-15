---
title: "Deconding From Language Models"
date: 2024-09-11
draft: false
---

## A quick refresher on Autoregressive text generation

{{< centered-image src="/images/decoding/decoding_1_a.png" alt="Alt text" >}}

Autoregressive language models generate text through a sequential process of predicting one token at a time. The model takes a sequence of tokens $ \lbrace y \rbrace _{<t} $ as input  and outputs a new token $ \hat{y_t} $. This process repeats iteratively, with each newly generated token becoming part of the input for the subsequent prediction.

At each time step $ t $, the model computes a vector of scores $ \mathbf{S} \in \mathbb{R}^V $, where $ V $ is the size of the vocabulary. For instance, in GPT-2, the vocabulary consists of 50,257 tokens.

$$
S = f(\lbrace y \rbrace _{<t} )
$$
<div style="text-align: center;">
where, \( f \) is the Language model.
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

{{< centered-image src="/images/decoding/decoding_1_b.png" alt="Alt text" >}}

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

{{< centered-image src="/images/decoding/decoding_1_c.png" alt="Alt text" >}}

Get the two most probable tokens, rest are pruned.

{{< centered-image src="/images/decoding/decoding_1_d.png" alt="Alt text" >}}

top 2 generations are “the poor” and “a poor”, these are kept and the reset are pruned

{{< centered-image src="/images/decoding/decoding_1_e.png" alt="Alt text" >}}

The above process is continued till we reach the \<EOS> end of sequence token or reach a limit for max output tokens.

{{< centered-image src="/images/decoding/decoding_1_f.png" alt="Alt text" >}}

Greedy decoding and Beam Seach are some of the most popular approaches for decoding in LLM’s like chat GPT, Llam and Gemini. 

### The issue with these approaches.

{{< centered-image src="/images/decoding/decoding_1_g.png" alt="Alt text" >}}

Open-ended text generation often leads to repetitive outputs. For example, when generating text about a unicorn trying to speak English, the continuation may initially appear coherent but soon start repeating phrases, like an institution's name, excessively. This repetition happens because the language model assigns increasing probability to repeated sequences, as shown in a plot of the model's probability for the sequence "I don't know." Initially, the probability is regular, but as the phrase repeats, the probability increases, indicating the model is more confident about the repetition.

{{< centered-image src="/images/decoding/decoding_1_h.png" alt="Alt text" >}}

This issue, known as self-amplification, persists even with larger models. For instance, models with 175 billion parameters still suffer from repetition when generating the most likely string. Increasing scale alone does not resolve this problem.
To mitigate repetition, one approach is n-gram blocking, which prevents the same n-gram from appearing twice. For example, if n is set to three, and the text contains "I am happy," the next time "I am" appears, "happy" would be set to zero probability, preventing the repetition of this trigram. However, n-gram blocking has limitations, as it can eliminate necessary repetitions, such as a person's name appearing multiple times.

{{< centered-image src="/images/decoding/decoding_1_i.png" alt="Alt text" >}}

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

{{< centered-image src="/images/decoding/decoding_2_a.png" alt="Alt text" >}}

#### Issues with ancestral sampling.

Sampling introduces a new set of challenges because it doesn't completely eliminate the probabilities of any tokens. In vanilla sampling, every token in the vocabulary remains a potential option, which can sometimes result in generating inappropriate words. Even with a well-trained model, where most of the probability mass is concentrated on a limited set of suitable options, the distribution's tail remains long due to the extensive vocabulary. This phenomenon, known as a heavy tail distribution, is characteristic of language. Consequently, when aggregated, the probabilities of these less likely tokens can still be significant. 

For instance, many tokens may be contextually inappropriate, yet a good language model assigns each a very small probability. However, the sheer number of these tokens means that, collectively, they still have a considerable chance of being selected. To address this issue, we can cut off the tail by zeroing out the probabilities of the unwanted tokens. 

### Top $ k $ Sampling

{{< centered-image src="/images/decoding/decoding_2_b.png" alt="Alt text" >}}

One effective approach is top-$k$ sampling, where we only sample from the top $k$ tokens in the probability distribution.
Increasing $k$ results in more diverse but potentially risky outputs, while decreasing $k$ leads to safer but more generic outputs.

{{< centered-image src="/images/decoding/decoding_2_c.png" alt="Alt text" >}}


Top-$k$ decoding can present two major issues. First, it can cut off too quickly, as illustrated by the sentence "She said, 'I never \___'." Many valid options, such as "won't" or "can't," might be excluded because they don't fall within the top $k$ candidates, leading to poor recall for the generation system. Second, top-$k$ decoding can also cut off too slowly. For instance, in the sentence "ate the pizza while it was still ___," the word "cold" is an unlikely choice according to common sense. Despite its low probability, the model might still sample "cold" as an output, resulting in poor precision for the generation model.

Given the problems with top-$k$ decoding, how can we address the issue that there is no single $k$ that fits all circumstances? The probability distributions we sample from are dynamic, and this variability causes issues. 
When the probability distribution is relatively flat, using a small $ k $ will eliminate many viable options, so we want $k$ to be larger in this case. Conversely, when the distribution is too peaky, a high $k$ would allow too many options to be viable. In this situation, we might want a smaller $k$ to be more selective.

### Top $P$ sampling (nucleus sampling)

The solution might be that $k$ is just a suboptimal hyperparameter. Instead of focusing on $k$, we should consider sampling from tokens within the top $P$ probability percentiles of the cumulative distribution function (CDF). This approach adapts to the shape of the probability distribution and can provide a more flexible and effective sampling method.


{{< centered-image src="/images/decoding/decoding_2_d.png" alt="Alt text" >}}

The advantage of using top-$P$ sampling, where we sample from the top $P$ percentile of the cumulative probability mass, is that it effectively gives us an adaptive $k$ for each different probability distribution. 

Let me explain what I mean by adaptive $k$. In the first distribution, which follows a typical power law of language, using top-$k$ sampling would mean selecting the top $k$ tokens. However, using top-$P$ sampling means we are focusing on the top $P$ percentile of the cumulative probability, which might be similar to top-K in this case.

For a relatively flat distribution like the blue one, top-$P$ sampling includes more candidates compared to top-$k$. On the other hand, for a more skewed distribution like the green one, top-$P$ sampling includes fewer candidates.
By selecting the top $P$ percentile in the probability distribution, we achieve a more flexible $k$, which better reflects the good options in the model. This adaptive approach ensures that we are considering an appropriate number of candidates based on the shape of the distribution, leading to more effective sampling.

### Epsilon Sampling

Epsilon sampling involves setting a threshold for lower bound probabilities. Essentially, if a word's probability is less than a certain value, like 0.03, that word will never appear in the output distribution. This ensures that low-probability words are excluded from the final output.

### Temperature Scaling

Another hyperparameter that we can tune to affect decoding is the temperature parameter $\( \tau \)$. Recall that at each time step, the model computes a score for each word, and we use the softmax function to convert these scores into a probability distribution:

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

{{< centered-image src="/images/decoding/decoding_2_e.png" alt="Alt text" >}}





Welcome to TensorTunes, where we explore the fascinating intersection of mathematics, music, and machine learning. Our goal is to uncover the hidden harmonies in data and sound, using the power of algorithms and signal processing.

Here's a simple example of how we might use LaTeX to express a musical concept mathematically:

The frequency $f$ of a note in the equal-tempered scale is given by:

$$
f = 440 \cdot 2^{\frac{n}{12}}
$$

Where $n$ is the number of semitones away from A4 (440 Hz).

In future posts, we'll dive deeper into topics like:

1. Fourier transforms and their applications in audio processing
2. Neural networks for music generation
3. Mathematical models of rhythm and meter
4. The geometry of musical scales and chords

Stay tuned for more exciting explorations at the crossroads of math and music!