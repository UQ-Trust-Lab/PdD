# PdD: The Ultimate Character-Level Perturbation Generator 🚀🔥

Welcome to **PdD**, the game-changing **Character-Level Perturbation Generator**! If you've ever wanted to push the boundaries of text data augmentation, you've come to the right place! PdD isn't just another tool—it’s your **text manipulation playground**, where every character can be twisted, swapped, and transformed to your heart’s content.

![image-20240123120135583](README.assets/image-20240123120135583.png)

## 🚀 Why PdD?

At its core, PdD is driven by three powerful principles: **P**robability distribution, **D**ensity, and **D**iversity. These metrics empower you to perturb text in ways that are dynamic, intelligent, and, most importantly, **impactful**! Whether you're enhancing neural network robustness, simulating real-world typos, or exploring creative text transformations, PdD gives you **full control** over your textual experiments.

## 🔥 Backed by Research

PdD is based on our paper “[Formalizing Robustness Against Character-Level Perturbations for Neural Network Language Models](https://link.springer.com/chapter/10.1007/978-981-99-7584-6_7).” Our work ensures that the perturbations aren’t just random noise but **structured, meaningful, and deeply insightful**. You can cite our work using the following BibTeX entry:

```bibtex
@inproceedings{ma2023formalizing,
  title={Formalizing Robustness Against Character-Level Perturbations for Neural Network Language Models},
  author={Ma, Zhongkui and Feng, Xinguo and Wang, Zihan and Liu, Shuofeng and Ma, Mengyao and Guan, Hao and Meng, Mark Huasong},
  booktitle={International Conference on Formal Engineering Methods},
  pages={100--117},
  year={2023},
  organization={Springer}
}
```

## 🛠️ How Does PdD Work?

Imagine this simple sentence:

> **"I am the best student in the world."**

Instead of processing whole words, PdD views this as a **sequence of characters**, unlocking powerful transformations:

```python
['I', ' ', 'a', 'm', ' ', 't', 'h', 'e', ' ', 'b', 'e', 's', 't', ' ',
 's', 't', 'u', 'd', 'e', 'n', 't', ' ', 'i', 'n', ' ',
 't', 'h', 'e', ' ', 'w', 'o', 'r', 'l', 'd', '.']
```

### 🎯 The Three Pillars of PdD

PdD gives you the **power** to customize perturbations with three key parameters:

- **🔄 Probability Distribution:** Control how characters are perturbed. Choose between **uniform** or **normal** distributions (managed in `distribution.py`).
- **📊 Density:** Define the fraction of characters to perturb. A density of `0.1` means **10% of characters** will be modified (adjustable in `sampler.py`).
- **🌀 Diversity:** The **fun** part! Select from a range of perturbations—deletion, typos, invisible characters, and more! Explore the `diversity` folder for all the possibilities.

## 🎭 Types of Perturbations You Can Apply

PdD isn’t just about adding random noise—it gives you full control over **how** text is perturbed. Here are some exciting options:

1. **✂️ Deletion** – Remove characters entirely (`''`).
2. **🔳 Space Insertion** – Turn letters into spaces (`' '`).
3. **🔀 Case Switching** – Flip uppercase and lowercase (`'a'` ↔ `'A'`).
4. **⌨️ Keyboard Typos** – Simulate real-world typos (e.g., `'g'` → `'h'`).
5. **🎭 Homoglyphs** – Replace characters with visually similar ones (e.g., `'a'` → `'а'`).
6. **🫥 Invisible Characters** – Insert invisible symbols that mess with models.

## 🚀 Get Started Now!

### 🔧 Installation

Make sure you're using **Python 3.10 or later** and install the required dependencies:

- torch
- tqdm
- scikit-learn
- transformers
- datasets

### 🏗️ File Structure Breakdown

PdD is structured to be intuitive and **easy to experiment with**:

- `./diversity` – The heart of PdD! Define custom perturbation rules here.
- `./example` – Quick-start examples to see PdD in action.
- `./scripts` – Scripts for running experiments based on our research.
- `./eval` – Code for training and evaluating models on perturbed data.
- `./results` – Stores original and perturbed outputs + evaluation results.

### 🎨 Customize Your Own Perturbations!

PdD is fully customizable. Want to tweak how certain characters get perturbed? Dive into the `diversity` folder and **define your own perturbation rules**. The power is in your hands! 💪

### 🎯 Target Specific Characters with Precision

Need to perturb only certain characters at specific locations? PdD lets you define exactly **where and how** to apply perturbations, giving you unparalleled control over text transformations.

---

## 💡 Join the Future of Text Manipulation!

With PdD, **text augmentation has never been this exciting**. Whether you're testing the robustness of AI models, mimicking human typos, or diving into linguistic experiments, PdD is here to help you explore **the untapped potential of character-level text transformations**.

Ready to revolutionize the way you handle text? **Let’s perturb some characters! 🔥**

