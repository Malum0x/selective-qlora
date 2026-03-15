# selective-qlora

Dataset optimization via perplexity filtering 

To improve finetuning efficiency, I implemented a script that scores dataset samples based on their cross-entropy loss using a pretrained model (qwen2.5-3B-Instruct). By selecting only the top 30% hardest samples, we reduce training noise and focus the models compute on non-trivial data, leading to better reasoning capabilities with less training time

While similar methods like Selective QloRA already exist, I am implementing this as a custom experimental test to observe firsthand how the model's behavior and training dynamics shift when exposed to a specificaly curated subset. Instead of relying on third-party benchmarks, this is a hands on evaluation of how the model performs on these specific examples after being filtered for difficulty. 

To ensure full transparency and rigous tracking, all experimental results, including loss curves and performance metrics, will be recorded and reported through the wandb irl. 


The filtered dataset (top 30% highest perplexity samples from OpenHermes 2.5) is publicly available on Hugging Face: [https://huggingface.co/datasets/Malum0x/openhermes2.5-Perplexity_filtered_top30]

After the first run (1 epoch), ARC Challenge shows a +0.5% improvement. A small margin, but training will continue for more epochs with validation loss tracking.
Using the same finetuning settings on the full OpenHermes dataset, ARC challenge performance is 8% lower than the top30% filtered version. 
tbc... 
