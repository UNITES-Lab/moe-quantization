# Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for the paper "**Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark**. Our codebase is built upon [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ).

Authors (* Equal Contribution): *[Pingzhi Li](https://pingzhili.github.io/)\*, [Xiaolong Jin](https://www.cs.purdue.edu/people/graduate-students/jin509.html)\*, [Yu Cheng](https://ych133.github.io/), and [Tianlong Chen](https://tianlong-chen.github.io/).* 



## Overview

Large Language Models (LLMs) have become foundational in the realm of natural language processing, demonstrating performance improvements as model sizes increase. The Mixture-of-Experts (MoE) approach offers a promising way to scale LLMs more efficiently by using fewer computational FLOPs through sparse activation. However, it suffers from significant memory overheads, necessitating model compression techniques. Post-training quantization, a popular method for model compression, proves less effective when directly applied to MoE models due to MoE's overlooked inherent sparsity. This paper explores several MoE structure-aware quantization heuristics, ranging from coarse to fine granularity, from MoE block to individual linear weight. Our investigations reveal critical principles: different MoE structures (i.e., blocks, experts, linear layers) require varying numbers of weight bits for effective and efficient quantization. Conclusions are supported by extensive benchmarking across two representative MoE models and six tasks. We further introduce novel enhancements to more accurately identify the most critical weights in MoE quantization that necessitate higher bit allocations, including the linear weight outlier scorer and MoE block scorer. Additionally, subsequent experiments validate our findings in the context of both weight and activation quantization.



## Getting Started

```bash
conda create -n qmoe python=3.10
conda activate qmoe
pip install -r requirements.txt
```



