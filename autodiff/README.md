# Toy Autodiff: A Simple Automatic Differentiation Library

Toy Autodiff is a minimalistic library to introduce the concept of automatic differentiation (autodiff). Autodiff is a powerful tool used in optimization, machine learning, and various scientific computations. The purpose of this toy library is educational; it provides a basic understanding of autodiff.
## Features
1. Scalar arithmetic including addition and multiplication
2. Computation of gradients (derivatives)
3. Simple backpropagation logic


## Structure
```plaintext
autodiff/                  # Root folder
├── autodiff/              # Sub-folder containing core code
│   ├── __pycache__/
│   ├── __init__.py
│   └── autodiff.py        # Core functionality
├── main.py                # Example usage or testing
├── pyproject.toml         # Dependency management with Poetry
└── README.md              # Documentation
```


## Usage

The `Scalar` class from the `autodiff` package provides functionalities for basic scalar operations and automatic differentiation. Below are some examples:

### Import the Scalar class

First, import the `Scalar` class:

```python
from autodiff import autodiff
```

### Basic Operations

You can initialize `Scalar` objects and perform basic operations like addition and multiplication:

```python
print(autodiff.Scalar(3.14))

print(autodiff.Scalar(3) + autodiff.Scalar(4) * autodiff.Scalar(5))
```

### Backpropagation with a Single Variable

Here, we initialize two Scalar objects `x` and `y`, both set to the same value:

```python
x = autodiff.Scalar(2.0)
y = x

x.grad = 0.0
y.grad = 1.0
y.backward()

print(x.grad)  # Should print the gradient of x after backpropagation
```

### Backpropagation with Addition

This example shows backpropagation involving addition:

```python
z = x + x
x.grad = 0.0
z.grad = 1.0
z.backward()

print(x.grad)  # Should print the gradient of x after backpropagation
```

### Backpropagation with Multiplication

This example demonstrates backpropagation with multiplication:

```python
a = autodiff.Scalar(3.0)
b = a * a

a.grad = 0.0
b.grad = 1.0
b.backward()

print(a.grad)  # Should print the gradient of a after backpropagation
```

### More Complex Function

This example calculates the gradient of a more complex function:

```python
k = autodiff.Scalar(3.0)
l = (k * k * k) + (autodiff.Scalar(4.0) * k) + autodiff.Scalar(1.0)

k.grad = 0
l.grad = 1
l.backward()

print(k.grad)  # Should print the gradient of k after backpropagation
```


