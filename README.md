# PdD: A Character-Level Perturbation Generator 🚀

![image-20240123120135583](README.assets/image-20240123120135583.png)

Welcome to **PdD**, the powerful **Character-level Perturbation Generator**! A framework is designed to manipulate text at the most granular level—each individual character!

Based on three core metrics, **P**robability distribution, **D**ensity, and **D**iversity, PdD allows you to transform and perturb text in dynamic ways. Whether you're working on enhancing neural network robustness or exploring creative text generation, PdD provides the tools to experiment with text like never before!

This project is based on our paper “[Formalizing Robustness Against Character-Level Perturbations for Neural Network Language Models](https://link.springer.com/chapter/10.1007/978-981-99-7584-6_7)”, and you can cite it as follows:

```tex
@inproceedings{ma2023formalizing,
  title={Formalizing Robustness Against Character-Level Perturbations for Neural Network Language Models},
  author={Ma, Zhongkui and Feng, Xinguo and Wang, Zihan and Liu, Shuofeng and Ma, Mengyao and Guan, Hao and Meng, Mark Huasong},
  booktitle={International Conference on Formal Engineering Methods},
  pages={100--117},
  year={2023},
  organization={Springer}
}
```



## What’s PdD All About?

Imagine a sentence like: "I am the best student in the world." Instead of thinking about words as a whole, PdD treats each sentence as a **list of individual characters**, opening up incredible possibilities for text manipulation!

For instance, the sentence above becomes:

```python
['I', ' ', 'a', 'm', ' ', 't', 'h', 'e', ' ', 'b', 'e', 's', 't', ' ', 's', 't', 'u', 'd', 'e', 'n', 't', ' ', 'i', 'n', ' ', 't', 'h', 'e', ' ', 'w', 'o', 'r', 'l', 'd', '.']
```

## The Magic Ingredients: Three Key Parameters

PdD gives you full control with **three exciting parameters** to fine-tune your perturbations!

- **Probability Distribution**: The chance of each character being perturbed. Choose between **uniform** or **normal** distribution (controlled in `distribution.py`).
- **Density**: This parameter determines how many characters are affected. For example, a density of `0.1` means that **10%** of characters will be perturbed! Set it anywhere between **0** and **1** (controlled in `sampler.py`).
- **Diversity**: The fun part! It defines the set of substitutions possible for each character. We've got a whole bunch of options, from **deleting characters** to replacing them with **keyboard typos**, **space**, **shifted keys**, and even **invisible characters**! Take a look at the `diversity` folder to explore these exciting choices.

## Types of Perturbations We Offer

1. **Deletion**: Replace a character with nothing (`''`).
2. **Space**: Turn a character into a space (`' '`).
3. **Letter Cases**: Switch a character’s case, e.g., `'a'` ↔ `'A'`.
4. **Shift-key**: Toggle between shifted and unshifted characters.
5. **Keyboard Typos**: Simulate typing mistakes by swapping a character with a nearby key.
6. **Homoglyphs**: Replace a character with a visually similar one (e.g., `'a'` → `'а'`).
7. **Invisible Characters**: Insert invisible characters after a letter to confuse your model!

## How to Get Started 🚀

The `main.py` file provides a simple yet powerful way to generate perturbed text. Just specify the three parameters—**Probability Distribution**, **Density**, and **Diversity**—and let PdD do the magic. The result? A perturbed version of your original sentence, ready for whatever you need!

## Customize Your Perturbation Dictionary

Want to get creative? You can easily customize your own perturbation dictionary by following the pattern found in the `diversity` folder. The possibilities are endless!

## Perturb Specific Characters at Specific Locations

Need more precision? You can preprocess your sentence to perturb specific characters at specified positions—perfect for more targeted text manipulation.
