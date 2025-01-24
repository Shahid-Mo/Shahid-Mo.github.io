+++
title = 'Knowledge distillation'
date = 2024-12-11
draft = true
math = true
+++

**Knowledge distillation**  is a compression technique in which a compact model - the student - is trained to reproduce the behaviour of a larger model - the teacher - or an ensemble of models.

In supervised learning, a classification model is trained to classify each sample by maximizing the estimated probability of the gold labels. A standard training objective thus involves minimizing the cross-entropy between the model’s predicted distribution and the one-hot empirical distribution of training labels. A model performing well on the training set will predict an output distribution with high probability on the correct class and with near-zero probabilities on other classes. But some of these "near-zero" probabilities are larger than others and reflect, in part, the generalization capabilities of the model and how well it will perform on the test set.

**Training loss**  
The student is trained with a distillation loss over the soft target probabilities of the teacher:  

$$
L_{ce} = \sum_i t_i \cdot \log(s_i)
$$

where $t_i$ (resp. $s_i$) is a probability estimated by the teacher (resp. the student). This objective results in a rich training signal by leveraging the full teacher distribution.  

Following Hinton et al. [2015], we used a **softmax-temperature**:

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

where $T$ controls the smoothness of the output distribution and $z_i$ is the model score for the class $i$. The same temperature $T$ is applied to the student and the teacher at training time, while at inference, $T$ is set to 1 to recover a standard **softmax**.

The final training objective is a linear combination of the distillation loss $L_{ce}$ with the supervised training loss, in our case the **masked language modeling loss $L_{mlm}$** [Devlin et al., 2018]. We found it beneficial to add a **cosine embedding loss ($L_{cos}$)** which will tend to align the directions of the student and teacher hidden states vectors.

