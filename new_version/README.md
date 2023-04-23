# Character-level Perturbation Generator

This is a project about character-level perturbation to text.

Each sentence is represented by a list of characters.

## Introduction

For example, the sentence "I love you." would be represented by the list `["I", " ", "l", "o", "v", "e", " ", "y", "o", "u", "."]`.

There are three main parameters that control the perturbation:

- **Distribution**: This parameter defines the probability distribution of all characters being perturbed. It define the probability of each character being perturbed. There are currently two choices: *uniform distribution* and *normal distribution*. The corresponding classes are defined in `distribution.py`.
- **Density**: This parameter controls how many characters will be perturbed, e.g., a density of $0.1$ means that $10\%$ of characters will be perturbed. This parameter should be a float between $0$ and $1$. It is controlled in `sampler.py`.
- **Diversity**: This parameter describes the possible substituted characters for each character. We provide different dictionaries for this parameter, which are located in the `diversity` folder. For each character, there is a list of possible substitutions. For example, "a" can be perturbed and replaced with "s" or "z"; therefore, the dictionary will have a key of `"a"` and a value of `["s", "z"]`. Currently, we have the following dictionaries.
  - **Deletion**: It replaces a character with an empty char `""`.
  - **Space**: It replaces a character with a space char `" "`.
  - **Letter cases**: It changes the cases of a letter, e.g. `"a"` to `"A"` or `"A"` to `"a"`.
  - **Shift-key**: It replaces a character with the corresponding key when using or not using the shift key. For example, `"3"` and `"#"`.
  - **Keyboard typos**: It replaces a character with the nearest character on a keyboard. For example, `"a"` and `"s"`.
  - **Homoglyphs**: It replace a character with its homoglyph. For example, `"a"` and` "$\alpha$"`
  - **Invisible Characters**: It insert an invisible chars, like zero width space, to after the original letter.

## Usage

The file `main.py` provides an example of how to perturb a sentence. You just need to specify the three parameters described above and build a generator. Then, you can obtain a perturbed version of a string.

### About customized perturbation dictionary

You can create your perturbation dictionary like what we do in the `diversity` folder.

### About perturbation to specified characters in specified position

Currently, all characters in the given sentence has a possibility to be perturbed. Therefore, If you want to perturb specified character on specified location, you need to pre-process the input sentence to pick out the target characters.



