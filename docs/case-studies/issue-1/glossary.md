# Glossary

Short, plain explanations of the terms used in this case study.

| Term | Meaning |
| --- | --- |
| **AUROC** | One number (0.5 = guessing, 1.0 = perfect) that shows how well a model separates sick from healthy cases at all thresholds. |
| **bf16** | A 16-bit number format. Each weight takes 2 bytes. It is the usual format for GPU inference and training. |
| **Calibration** | Whether the model's confidence matches reality: answers given with "90% sure" should be right 90% of the time. |
| **CER / WER** | Character / word error rate. The share of characters (or words) that OCR gets wrong. 0 is perfect. |
| **cgroup** | A Linux limit on memory and CPU for a group of processes. Our dev VM has 12.5 GB of RAM but a 3 GiB cgroup limit. |
| **Confidence interval (95%)** | The range where the true score most likely lies. With few test items it is wide, so two models can only be called different if their ranges do not overlap much. |
| **Continued pre-training** | More training of a ready model on a large pile of unlabeled text or images from a new field (here: medicine). |
| **Decode / generation speed** | How many answer tokens a model writes per second. On CPU it is limited by memory speed. |
| **Differential diagnosis** | A ranked list of possible diseases that could explain the symptoms. |
| **Distillation** | A small "student" model learns to copy the answers of a big "teacher" model. |
| **DPO** | Direct Preference Optimization. Training on pairs of answers where people marked which one is better. |
| **Fine-tuning** | More training of a ready model on a small labeled dataset for one task. |
| **Flow matching** | A way to generate images: the model learns how to move random noise step by step towards a real image. UniMedVL uses it. |
| **GGUF** | The file format of llama.cpp. It stores weights, often in 4-bit form, plus the tokenizer. |
| **GRPO** | Group Relative Policy Optimization. Reinforcement learning where the model writes several answers, a checker scores them, and the model learns from the better ones. No separate reward model is needed. |
| **Image tokens** | The vectors an image becomes inside a VLM. The language model reads them like words. MedGemma uses 256 per image. |
| **KV cache** | Memory where the model keeps what it has already read, so it does not recompute it. It grows with the prompt length. |
| **LoRA / QLoRA** | Low-Rank Adaptation. Training only small extra matrices (under 1% of the weights) instead of the whole model. QLoRA does the same on top of a 4-bit base model, so it needs much less GPU memory. |
| **mmap** | Reading model weights straight from the file on disk instead of copying them into RAM. It saves memory but becomes very slow when RAM is too small to cache the file. |
| **MoT (Mixture-of-Transformers)** | A design with several full copies of the transformer layers. Each token goes to one copy by its type (text, image to understand, image to generate), and all copies share attention. UniMedVL uses it. |
| **MRR** | Mean reciprocal rank. 1 if the true diagnosis is first, 0.5 if second, 0.33 if third, and so on, averaged over cases. |
| **NaViT** | A vision encoder that keeps the image's own size and shape instead of squeezing it into a square. |
| **OCR** | Optical character recognition: turning a picture of text into text. |
| **Patch** | A small square of an image (for example 14×14 pixels). The vision encoder turns each patch into a vector. |
| **Phantom** | A made-up image that imitates a real scan. Ours are drawn by code, so we know exactly where the "finding" is. |
| **Prompt processing (prefill) speed** | How many tokens of the question a model reads per second before it starts answering. Images add hundreds of tokens, so this matters for VLMs. |
| **Quantization (Q4_K_M, nf4)** | Storing weights with fewer bits, for example about 4.9 bits per weight for Q4_K_M instead of 16. The model becomes 3–4 times smaller with a small loss of quality. |
| **RAG** | Retrieval-augmented generation. Looking up relevant documents (guidelines, drug databases) and adding them to the prompt. |
| **Reasoning model** | A model trained to "think" in text before giving the final answer. |
| **Reinforcement learning (RL)** | Training by rewarding good answers instead of showing the exact right text. |
| **RxNorm** | The US National Library of Medicine list of drug names and strengths. We use it to check drug names read by OCR. |
| **Self-hosted runner** | A machine you own that runs GitHub Actions jobs, for example one with a GPU. |
| **Self-supervised learning** | Learning from unlabeled data, for example by hiding part of an image and predicting it. |
| **Sensitivity / specificity** | Sensitivity: the share of sick cases the model finds. Specificity: the share of healthy cases it correctly calls healthy. |
| **Synthetic data** | Data made by a program (Synthea patients, rendered prescriptions, phantoms) instead of collected from real patients. |
| **Token** | A piece of text (about ¾ of an English word) or one image vector. Models read and write tokens. |
| **top-k accuracy** | The share of cases where the true answer is among the model's first *k* answers. |
| **Transformer** | The neural network design behind modern language models. Its layers use *attention*, so every token can look at the others. |
| **Vision encoder** | The part of a VLM that turns pixels into vectors (for example SigLIP or MedSigLIP). |
| **VLM** | Vision-language model: a model that reads images and text and writes text. |
