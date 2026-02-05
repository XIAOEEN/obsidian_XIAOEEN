旨在识别评价对象（实体），及其特定属性维度（层级）的情感倾向。细粒度的文本情感分类。

1. 任务形式定义：
	- 一个完整的观点通常被定义为一个五元组（e，A, s, h, t）:
	- Entertity, Ascpect, semtiment, holder,time
2. 主要子任务：
	1. 属性层级抽取（AE，Aspect-Extraction，识别属性词）
	2. 观点抽取（OE， Opinion Extrraction，识别情感表达词）
	3. 属性级情感分类（ALSC，Aspect-level Sentiment classification,针对特定属性，判断具体的情感标签）
		1. 联合抽取任务：如属性-观点对抽取、属性-观点-情感三元组抽取