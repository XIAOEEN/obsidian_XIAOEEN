- ***keywords: Sentence-level, world-wise, Detection of AIGT***
- ***Date to read: 2026-01-21***
### Motivation:
#### BackGround side:

 The current AIGT methods face two main challenges:
 1. model-wise mehtods like DetectGPT, sniffer require a long document as input text (over 100 tokens), making them less effective to detect short text.
 
 2. supervised methods like fine-tuning RoBERTa prone to overfitting on the training data.
 
 3. The document-level detection methods struggle to accurately evaluate documents containing a mix of AI-generated and human-authored content or consisting of few sentences, which can potentially lead to a higher rate of false negatives or false positives
Therefore, it is necessary to ***build a sentence level AIGT detecctors*** to deal with the challenges aboved.


#### Methods Theoretical side:

1. They found perplexity is a significant feature to be used in AIGT detectors.[1]

2. They think the world log probability list are composed  like waves in speech processing where convolution network are often be used. So they use conv network to extract this wave feature[2], followed by a self-attention layers, to process wave like features.
主要来源于sniffer 和Detect GPT，Previous works demonstrate that both the average per-token log probability (DetectGPT) and contrastive features derived from token-wise probabilities (Sniffer) can contribute to AIGT detection.因此他们选择去提取token-wise log probability  list


### 主要实现方案：
![[截屏2026-01-21 14.10.28.png]]

1. 卷积神经网络对world-wise probability list 卷积提取foundation-feature(convalution network output channel (64, 128*3, 64))
2. concat四个代理模型输出的foundation-feature, 送入transformer的encoder（2 layers self-attention, each 16heads, 256 dimension FFN），获取Contextualized Features.
3. 最后送入Linear CLassification Layer, output put world-wise label. 最后聚合world-wise label to sent label and context label.

### About DataSets:
1.  Particular-Model Binary Detection Dataset
2.  Mixed-Model Binary Detection Dataset
3.  Mixed-Model Multiclass  Detection Dataset 

Specially， 1，2 construct for Binary Detection Task。3 constrcut for MutiClass Detection Task, which means not only discriminate AI or human Text, but also figure out the origins(LLMs,GPT, Claude, etc) of the Text 

### Main Results:

![[截屏2026-01-21 15.10.03.png]]

### About ppl&nll

https://www.cnblogs.com/GraphL/p/18387519

NLL: native log-likelihood,

1. 计算 每一个token的logits

2. 对logits list 做softmax得到 $P_{i-token}$
	 $$softmax: p_i =\frac{e^{zi}}{\sum{e^{zj}}}$$

4. - log($P_{targetToken}$) 就是负似然对数NLL，这里$P_{targetToken}$是指真实token的概率

PPl: perplexity 困惑度
- ppl = $e^{AvgNLL}$

- AvgNLL = $\frac{1}{n} \sum_{i=1}^n\ell_{i}$ , $\ell_i =- log(P_{i-token})$

- 因此对于任一给定的sequence的ppl：所有token负似然对数损失和的平均

这里补充一个交叉熵损失函数的概念：
$$L = - \sum_{i}^CP_ilog(\hat P_{i})$$
这里的$P_i$真实值标签的概率（一般是ont hot编码）$\hat P_{i}$ 预测标签的概率（一般是一个概率列表，连续值）

对于语言模型来说，因为真实标签只有一个，因此交叉熵可以简化：
$$L = - log(P_{targetToken})$$
因此对于语言模型来说，负对数似然，其实就是交叉熵损失。对于一个序列来说，ppl 是 交叉熵损失和的平均，取指数。



### 参考文献

[1]Sebastian Gehrmann, Hendrik Strobelt, and Alexander Rush. 2019. GLTR: Statistical detection and visualization of generated text. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 111–116, Florence, Italy. Association for Computational Linguistics.

[2]Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460.



