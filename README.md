# selective-qlora

Dataset optimization via perplexity filtering 

To improve finetuning efficiency, I implemented a script that scores dataset samples based on their cross-entropy loss using a pretrained model (qwen2.5-3B-Instruct). By selecting only the top 30% hardest samples, we reduce training noise and focus the models compute on non-trivial data, leading to better reasoning capabilities with less training time

While similar methods like Selective QloRA already exist, I am implementing this as a custom experimental test to observe firsthand how the model's behavior and training dynamics shift when exposed to a specificaly curated subset. Instead of relying on third-party benchmarks, this is a hands on evaluation of how the model performs on these specific examples after being filtered for difficulty. 

To ensure full transparency and rigous tracking, all experimental results, including loss curves and performance metrics, will be recorded and reported through the wandb irl. 


tbc... 
