

## [PAPER 265] VERITAS: Veracity-Enhanced Robust Identification of LLM-generated Text Against Style-shifts

**1. Brief Summary**
The paper addresses a critical vulnerability in current large language model (LLM) generated text detectors: their susceptibility to style-based adversarial attacks. The authors point out that attackers can evade detection by prompting LLMs to change the text style, leading to a significant decrease in the accuracy of existing detectors, by as much as 16.45%. To address this problem, the paper proposes the VERITAS framework, a style-agnostic detection method. This method comprises three core components: (1) data augmentation using an adversarially enriched dataset generated through LLM fine-tuning; (2) a training paradigm incorporating both Style Alignment Loss and classification loss to decouple content semantics from style variations; and (3) content-based attribution supervision, utilizing LLMs to extract features such as semantic consistency and logical coherence. Experimental results demonstrate that VERITAS significantly outperforms existing state-of-the-art methods such as DetectGPT and GLTR on the Story, PolitiFact, and Science datasets, and against various style attacks.

**2. Strengths**

1. The paper astutely points out the shortcomings of current detectors that overly rely on static linguistic features such as vocabulary choice and syntactic patterns. This article defines clear attack scenarios (such as narrative, political, and scientific style reconstruction), making it highly valuable for research.
2. This paper was tested on the Story, PolitiFact, and Science datasets, and compared against various baseline models including DetectGPT, GLTR, SeqXGPT, and general LLMs (such as GPT-4 and Deepseek-R1 ), with comprehensive ablation studies. Furthermore, the paper conducted cross-domain evaluation, demonstrating that models trained solely on the Story dataset still maintain high F1 scores (80.16%-83.30%) on the Science dataset, showcasing excellent generalization capabilities.
    

**3. Weaknesses**

1. To fully demonstrate the advantages of the VERITAS framework, the authors are encouraged to include one or more detection frameworks that also utilize contrastive learning and representation learning (such as Detective[1]) for comparison. Since these models already consider the diversity of generated styles in their design, comparing their performance degradation under style adversarial perturbations  will more clearly and convincingly define VERITAS's technical innovations and practical performance gains.
    
2. VERITAS's data augmentation and attribution supervision both heavily rely on the capabilities of auxiliary LLMs (used for generating style variations and extracting attributions). If the auxiliary LLM itself has biases in understanding certain styles, it may introduce noisy labels. The paper does not delve into a detailed sensitivity analysis of how the choice of different auxiliary LLMs (e.g., using GPT-4 vs. Llama-2) specifically impacts the final detector performance.

3. Although the proposed VERITAS framework is conceptually sound, its empirical improvements over existing strong baselines appear relatively limited. This raises concerns about the practical significance of the method. In particular, under certain style-based attack configurations, VERITAS's performance is comparable to existing robust detectors (Table 2, e.g., only a 0.29% improvement for story-D compared to RoBERTa-finetuned (ACL 2024)), but not significantly better, leading to doubts about whether the improvement stems from advantages in the training dataset rather than the superiority of the method itself.
    
[1]Guo, Xun, et al. "Detective: Detecting ai-generated text via multi-level contrastive learning." _Advances in Neural Information Processing Systems_ 37 (2024): 88320-88347.


**4.Relevance to Web4Good**

This paper is related to the Web4Good theme. The misuse of LLM-generated text is a significant threat to the current cyberspace, particularly in creating fake news, misleading public opinion, and academic misconduct. By providing a robust detection tool capable of resisting style mimicry, this research contributes to purifying the online environment and maintaining the authenticity of information.

**5.Ethical Issues**

None.

**Rating:**

 weak reject
