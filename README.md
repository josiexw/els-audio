# 9.520 Project

**Large Embedding Models (LEMs) and Memory Compression/Decompression**

This project explores the hypothesis that intelligence may have originated from mechanisms akin to diffusion models, to reconstruct full memories from noisy or partial fragments, before the advent of language. We will examine the idea that Large Embedding Models (LEMs)—systems that can recover complete embeddings from incomplete or compressed versions—represent a step older than today’s LLMs. The biological analogy is that evolution discovered “thinking with embeddings” (e.g., memory recall, dreams, and imagination) before language.

The project consists of  investigating this hypothesis through computational experiments. A suggested starting point is to simulate “fragmented memories” by compressing or reducing the bit depth of image embeddings, then train simple networks (e.g., shallow recurrent or feedforward architectures) to reconstruct the original embedding or the full image. Alternatively mask part of original images and try to reconstruct from fragments. Theoretical and empirical connections between compression, learning, and creativity should be considered: Can reconstruction from fragments emulate aspects of human memory recall or dream generation?

Possible extensions include relating the results to the work of Kamb and Ganguli on creativity, or to recent theories of AI and human imagination as “filling in the gaps” from incomplete information. Students are encouraged to test the limits of fragment-based learning, compare autoregressive vs. diffusion-style reconstruction, and reflect on what such models suggest about the origins of thought.

---
**Overview**
This project is an extension on An analytic theory of creativity in convolutional diffusion models by Kamb and Ganguli.
We explore whether their ELS machine for creativity in convolutional diffusion models applies to modalities other than
images. In this case, we implement the ELS machine for audio, comparing its predictions with audio diffusion model outputs.

---

## Modules

```
9520_project/
│
├── assets/
│
├── checkpoints/
│   └── ddpm/
│
├── data/
│   └── esc50/
│       ├── test/
│       └── train/
│
├── src/
│   ├── data_preprocessing/
│   │   ├── complex_morlet_transformation.py
│   │   ├── data_split.py
│   │   └── generate_patch_bank.py
│   │
│   └── models/
│       └── ddpm.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Downloading dataset
- [ESC-50](https://github.com/karolpiczak/ESC-50)

## Installation
```bash
https://github.com/josiexw/els-audio.git
```
