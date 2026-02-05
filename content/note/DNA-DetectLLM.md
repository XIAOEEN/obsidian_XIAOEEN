Keywords： mutation-repair paradigm into AI-generated text detection. zero-shot Detection Method
### Motivation：
In molecular biology, DNA’s double-helix structure ensures stable transmission of genetic information, yet mutations during replication introduce variations that can lead to individual differences or even diseases such as cancer.
DNA的双螺旋结构包含了一个稳定的基因信息，但是在基因复制的过程中一个突变可能会引发变异，导致疾病或者癌症

an ideal AI-generated text sequence can be seen as a “template strand”, representing the most probable token choices at each position. Human-written texts, by contrast, ***resemble mutated strands, where token selections deviate from the optimal probabilities, creating measurable differences.***

一个理想的AIGT 可以被视为一种标准的基因模版，每一个位置的token都是最佳选择（最高概率），但是人类文本，类似于突变的链，有些位置token的选择偏离了最优选择。

**核心方法：by analogizing to DNA base-repair processes, we iteratively “correct” non-optimal tokens in a text and measure the difficulty of restoring it to the ideal AI-generated form. This repair-based approach captures the intrinsic divergence between AI-generated and human-written texts in a direct and interpretable manner.
核心点就是人类文本，有些token是非概率最高的token，本质上与基于ppl的方法或者基于扰动的方法无区别。

![[截屏2026-01-21 17.01.33.png]]

### Methods：
***How to perform a token-by-token repair process on the input sequence ?***

简单来讲就是，input same text $S_{taget}$,  to $M_s$ , and output a predict sequence $S_1$.
1. 逐token比较$S_{target}$ 和$S_q$ , 定位mutation token, 然后再拿去过一遍$M_s$ , 这次只采样mutation token的位置，其他位置保持不变，迭代这个操作直到所有的位置都和$S_target$ sequence 的token完全一样.
2. 用原文里的公式来表达这个过程：$$x_i \in s = \{x_1, x_2,..., x_L\} \to \hat x_i = \arg \max_{\tilde x \in V} P_{M_s}(\tilde x | x_{<i}), \quad if \, x_i \neq \hat x_i $$
***How to conculate repair scores?***

简单来说就是平均迭代的ppl差值：
$$Repair \,\,Scores = LogPPL_{M}(S) - LogPPL_{M}(\tilde S) $$ where $S$ indicates a given origin text, $\tilde S$ indicates a repaired text.
上面的式子也等价于：
$$\frac{1}{T} \sum_{i}^{T}[ log{P_M}​(\tilde x_i​∣\tilde x_{<i}​)−logPM​(x_i​∣x<i​)]$$


![[截屏2026-01-21 17.32.56.png]]