# Align Beyond Prompts: Evaluating World Knowledge Alignment in Text-to-Image Generation




<img src="./assets/figure1.png" width=1000px>

Recent text-to-image (T2I) generation models have advanced significantly, enabling the creation of high-fidelity images from textual prompts. However, existing evaluation benchmarks primarily focus on the explicit alignment between generated images and prompts, neglecting the alignment with real-world knowledge beyond prompts. To address this gap, we introduce Align Beyond Prompts, a comprehensive benchmark designed to measure the alignment of generated images with real-world knowledge that extends beyond the explicit user prompts. ABP comprises over 2,000 meticulously crafted prompts, covering real-world knowledge across six distinct scenarios. We further introduce ABPScore, a metric that utilizes existing Multimodal Large Language Models (MLLMs) to assess the alignment between generated images and world knowledge beyond prompts, which demonstrates strong correlations with human judgments. Through a comprehensive evaluation of 8 popular T2I models using ABP, we find that even state-of-the-art models, such as GPT-4o, face limitations in integrating simple commonsense knowledge into generated images. 
To mitigate this issue, we introduce a training-free strategy within ABP, named Inference-Time Knowledge Injection (ITKI). By applying this approach to optimize 200 challenging samples, we achieved an improvement of approximately 43% in ABPScore.

Due to the large size, generated images are stored separately on <a href="https://huggingface.co/datasets/smileying/ABP" target="_blank">Hugging Face</a>.


<br>

## Setup

```bash
git clone https://github.com/smile963/ABP.git
cd ABP
pip install -r requirements.txt
```

## Benchmarking Text-to-Image Models 

```bash
python scripts/t2i_eval_ABP.py
```

## Human correlation

```bash
python human_annotations/compute_correlation.py
```

## Analysis

### Different T2I models’ results on ABP

| **Models**    | **Physical Scenes** | **Chemical Scenes** | **Animal Scenes** | **Plant Scenes** | **Human Scenes** | **Factual Scenes** | **Overall** |
| ------------- | ------------------- | ------------------- | ----------------- | ---------------- | ---------------- | ------------------ | ----------- |
| SDXL          | 0.6511              | 0.5283              | 0.6282            | 0.6924           | 0.6857           | 0.7489             | 0.6558      |
| SD3-M         | 0.7011              | 0.5647              | 0.6257            | 0.6923           | 0.7073           | 0.7528             | 0.6740      |
| SD3.5-L       | 0.7091              | 0.5734              | 0.6656            | 0.7259           | 0.7226           | 0.7787             | 0.6959      |
| CogView4      | 0.7205              | 0.6228              | 0.6215            | 0.7132           | 0.7201           | 0.8039             | 0.7003      |
| Midjourney V6 | 0.7153              | 0.5843              | 0.7219            | 0.7553           | 0.7360           | 0.8123             | 0.7208      |
| Gemini 2.0    | 0.7397              | 0.6626              | 0.7129            | 0.7371           | 0.7528           | 0.7753             | 0.7301      |
| DALL-E 3      | 0.7630              | 0.7107              | 0.7738            | 0.8077           | 0.7463           | 0.8346             | 0.7727      |
| **GPT-4o**    | **0.8180**          | **0.7702**          | **0.8243**        | **0.8421**       | **0.8152**       | **0.8581**         | **0.8213**  |

### Correlations between each evaluation metric and human judgment on ABP.

| **Method**          | **Spearman's ρ** | **Kendall's τ** |
| ------------------- | ---------------- | --------------- |
| CLIPScore           | 11.2             | 7.5             |
| SigLIP              | 16.6             | 10.9            |
| HPS V2              | 10.6             | 7.1             |
| ImageReward         | 17.0             | 10.9            |
| PickScore           | 19.1             | 12.9            |
| SCISCORE            | 16.1             | 11.1            |
| **ABPScore (Ours)** | **43.4**         | **32.3**        |



## More details are coming soon...
