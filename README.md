# Perplexity weighted selective finetuning 

Dataset optimization via perplexity filtering.

## Overview

To improve fine-tuning efficiency, I implemented a script that scores 
dataset samples based on their cross-entropy loss using a pretrained 
model (Qwen2.5-3B-Instruct). By selecting only the top 30% hardest 
samples, we reduce training noise and focus the model's compute on 
non-trivial data, leading to better reasoning capabilities with less 
training time.

While similar methods already exist, this implementation serves as a 
custom experiment to observe firsthand how model behavior and training 
dynamics shift when exposed to a specifically curated subset. Rather 
than relying solely on third-party benchmarks, the evaluation focuses 
on direct observation of how the model responds to difficulty-filtered 
data — with ARC Challenge and GSM8K used as external reference points.

To ensure full transparency and rigorous tracking, all experimental 
results including loss curves and performance metrics are recorded 
through W&B.

The filtered dataset (top 30% highest perplexity samples from 
OpenHermes 2.5) is publicly available on Hugging Face:
https://huggingface.co/datasets/Malum0x/openhermes2.5-Perplexity_filtered_top30

## Results

| Model               | ARC-Challenge | GSM8K |
|---------------------|---------------|-------|
| Base Model          | 82.0%         | 68.0% |
| Baseline Fine-tuned | 80.5%         | 60.0% |
| Filtered Fine-tuned | 82.5%         | 61.0% |

## Findings

After a 3-epoch run, results showed no meaningful improvement over 
1 epoch. The dataset may be too small, or the learning rate too high 
— the model shows signs of catastrophic forgetting rather than 
meaningful adaptation.

Standard fine-tuning on full OpenHermes degrades model performance 
across both benchmarks. Perplexity-based difficulty filtering 
partially mitigates this — recovering ARC Challenge performance above 
the base model — but math reasoning (GSM8K) remains degraded.

Perplexity filtering alone is insufficient to protect math reasoning 
capabilities. This suggests that layers responsible for mathematical 
reasoning are being overwritten during fine-tuning regardless of data 
quality. Gradient-norm targeted layer selection may preserve these 
capabilities by avoiding unnecessary updates to math-critical layers.

*This finding motivates the next project: gradient-norm based 
selective layer fine-tuning.*



## wandb baseline training (full openhermes finetuning): 




