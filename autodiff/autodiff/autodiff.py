class Scalar:
    """A class to represent a scalar value and its gradient for automatic differentiation.

    Attributes:
        val (float): The actual scalar value.
        grad (float): The gradient (derivative) of the scalar. Initialized to 0.0.
        backward (function): A lambda function for backpropagation. Initialized to a no-op.

    Methods:
        __add__(self, other): Overloads the + operator to enable addition of two Scalar objects.
        __mul__(self, other): Overloads the * operator to enable multiplication of two Scalar objects.
    """
    def __init__(self, val) -> None:
        """Initializes the Scalar with a given value.

        Args:
            val (float): The actual scalar value.

        Returns:
            None
        """
        self.val = val
        self.grad = 0.0
        self.backward = lambda: None

    def __repr__(self) -> str:
        """Returns a string representation of the Scalar object.

        Returns:
            str: A string showing the value and gradient.
        """
        return f"Value: {self.val}, Gradient: {self.grad}"

    def __add__(self, other):
      """Overloads the + operator for Scalar addition and defines backpropagation logic.

        Args:
            other (Scalar): Another Scalar object.

        Returns:
            Scalar: A new Scalar object representing the sum.
        """
        
        out = Scalar(self.val + other.val)

        def backward():
            self.grad += out.grad
            other.grad += out.grad
            self.backward()
            other.backward()

        out.backward = backward

        return out

    def __mul__(self, other):
       """Overloads the * operator for Scalar multiplication but defines backpropagation logic.

        Args:
            other (Scalar): Another Scalar object.

        Returns:
            Scalar: A new Scalar object representing the product.
        """
        out = Scalar(self.val * other.val)

        def backward():
          self.grad += out.grad * other.val
          other.grad += out.grad * self.val
          self.backward()
          other.backward()
        
        out.backward = backward
        
        return out
