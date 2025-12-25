# Vectors, Matrices, and Tensors

## Core definitions

**Vector**  
A vector is an ordered list of numbers that represents a point or direction in space, and in machine learning it typically represents features or an embedding.

**Matrix**  
A matrix is a rectangular grid of numbers that represents a linear transformation or a collection of vectors stacked together.

**Tensor**  
A tensor is a generalization of vectors and matrices to higher dimensions, used to represent structured, multi-dimensional data.

---

## Why these objects matter in AI

Most of the mathematics behind modern machine learning models is built from vectors, matrices, and tensors.  
Embeddings, model parameters, intermediate activations, and gradients are all expressed using these objects.  
Understanding how they behave and how their shapes interact is essential for reasoning about models.

---

## Vectors

### Intuition
A vector can be viewed as:
- a list of features
- a point in space
- a direction with magnitude

These interpretations are mathematically equivalent and appear in different ML contexts.

### Notation and shape
A vector in $\mathbb{R}^n$ contains $n$ real numbers:

$$
\mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

- Shape: `(n,)` or `(n, 1)`
- Commonly represented as a one-dimensional array in code

### Example
$$
\mathbf{x} = [2,\; -1,\; 0.5]
$$

This vector could represent input features, an embedding, or a gradient, depending on context.

---

## Matrices

### Intuition
A matrix can be interpreted as:
- a collection of vectors stacked together, or  
- a linear function that maps one vector to another

In machine learning, matrices most often represent linear transformations.

### Notation and shape
A matrix with $m$ rows and $n$ columns is written as:

$$
A \in \mathbb{R}^{m \times n}
$$

Example:

$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

- Shape: `(2, 3)`
- Interpreted as mapping a 3-dimensional input vector to a 2-dimensional output vector

### Matrix–vector multiplication
If:

$$
A \in \mathbb{R}^{m \times n}, \quad \mathbf{x} \in \mathbb{R}^n
$$

then:

$$
\mathbf{y} = A\mathbf{x} \in \mathbb{R}^m
$$

This operation is the core computation performed by a linear layer in a neural network.

### ML connection
A standard linear model is written as:

$$
\mathbf{y} = W\mathbf{x} + \mathbf{b}
$$

where:
- $W$ is a weight matrix
- $\mathbf{x}$ is the input vector
- $\mathbf{y}$ is the output vector
- $\mathbf{b}$ is a bias vector

---

## Tensors

### Intuition
A tensor extends vectors and matrices to multiple dimensions.  
Each axis of a tensor has a specific semantic meaning, such as batch size, feature dimension, or spatial location.

### Common tensor shapes in ML

| Example | Shape | Interpretation |
|------|------|----------------|
| Batch of vectors | `(batch_size, d)` | multiple feature vectors |
| Sequence of embeddings | `(seq_len, d)` | ordered token embeddings |
| Batch of sequences | `(batch_size, seq_len, d)` | NLP model inputs |
| Image batch | `(batch_size, channels, height, width)` | image data |

Mathematically, tensors do not introduce new operations; they organize vectors and matrices in structured ways.

---

## Shapes as a first-class concept

Correctly reasoning about tensor shapes is often more important than memorizing formulas.  
Shape mismatches are a common source of implementation errors in machine learning systems.

Key questions to ask:
- What does each axis represent?
- Which dimensions must align for an operation?
- What shape should the output have?

---

## Small concrete examples

### Vector
$$
\mathbf{x} = [1, 2, 3]
$$

- Shape: `(3,)`
- Represents three numerical features

### Matrix
$$
W =
\begin{bmatrix}
1 & 0 & -1 \\
2 & 1 & 0
\end{bmatrix}
$$

- Shape: `(2, 3)`
- Maps a 3-dimensional vector to a 2-dimensional vector

### Tensor
A collection of 4 vectors, each of dimension 3:
- Shape: `(4, 3)`
- Represents a batch of inputs

---

## Minimal NumPy example

```python
import numpy as np

# vector
x = np.array([1.0, 2.0, 3.0])  # shape (3,)

# matrix
W = np.array([
    [1.0, 0.0, -1.0],
    [2.0, 1.0,  0.0]
])  # shape (2, 3)

# matrix-vector multiplication
y = W @ x  # shape (2,)

print(y)
