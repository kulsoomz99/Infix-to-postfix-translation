# Infix-to-Postfix Translation using Seq2Seq with Attention

##
- Kulsoom Zaidi

This project implements a neural network-based approach to translate mathematical formulas from traditional **infix notation** (e.g., `(a + b) * c`) to **postfix notation** / Reverse Polish Notation (e.g., `ab+c*`). 

Instead of relying on hardcoded algorithmic rules (like the Shunting Yard algorithm), this project uses a data-driven **Sequence-to-Sequence (Seq2Seq) Encoder-Decoder architecture with an Attention Mechanism** to learn the underlying syntactic rules, operator precedence, and parenthetical groupings.

---

## Problem Description

Infix notation is the standard for human-readable mathematics but is inherently ambiguous without parentheses or strict precedence rules. Postfix notation eliminates this ambiguity by explicitly encoding the order of operations through the position of operators relative to operands, making it ideal for stack-based evaluation and programmatic parsing.

**Example:**
* **Infix:** `( ( a + b ) * c )`
* **Postfix:** `a b + c *`

The goal of this project is to train a neural network to accurately perform this disambiguation and translation autoregressively.

---

## Model Architecture

The core model is a **Seq2Seq Encoder-Decoder with Attention**, built using TensorFlow/Keras.

* **Encoder:** 
  * Embedding Layer $\rightarrow$ Bi-directional/Multi-layer LSTM (128 dimensions).
  * Reads the tokenized infix expression and generates a sequence of hidden states.
* **Decoder:** 
  * Embedding Layer $\rightarrow$ LSTM (128 dimensions).
  * **Attention Mechanism:** Allows the decoder to focus on relevant parts of the input sequence when generating each output token.
  * **Output Layer:** Dense layer with Softmax activation to predict the next token probability distribution.
* **Decoding Strategy:** Greedy Autoregressive Decoding (Beam search is explicitly disabled per project constraints).

### Model Constraints & Specs
* **Max Parameters:** < 2,000,000 (Actual Trainable Params: ~271,376)
* **Max Syntactic Depth:** 4 (Abstract Syntax Trees have at most 4 levels)
* **Vocabulary:** Identifiers (`a-f`), Operators (`+, -, *, /`), Parentheses, and Special Tokens (`PAD`, `SOS`, `EOS`).

---

## Evaluation Metric: Prefix Accuracy

Because exact sequence matching can be overly penalizing for long sequences, the model is evaluated using **Prefix Accuracy**. 

This metric measures the length of the initial prefix of the predicted sequence (`y_pred`) that perfectly matches the ground truth (`y_true`), divided by the maximum length of the two sequences (up to the `EOS` token).
* **Score of 1.0:** Perfect match.
* **Score of 0.5:** The model got the first half of the expression correct before making an error.

---

## Results

The model was evaluated on a test set of 30 randomly generated expressions, repeated over 10 rounds to ensure stability.

| Metric | Mean | Standard Deviation |
| :--- | :---: | :---: |
| **Prefix Accuracy** | **1.0000** | **0.0000** |

*Conclusion:* The model achieved a perfect 100% prefix accuracy with zero variance across all evaluation rounds, demonstrating that the attention mechanism highly effectively maps complex symbolic sequences and perfectly learns the rules of operator precedence and grouping within the bounded depth limit.

---

## Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

### Dependencies
Install the required packages via `pip`:
```bash
pip install tensorflow numpy pandas matplotlib gdown
