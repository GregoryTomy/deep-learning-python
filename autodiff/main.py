from autodiff import autodiff

print(autodiff.Scalar(3.14))

print(autodiff.Scalar(3) + autodiff.Scalar(4) * autodiff.Scalar(5))

x = autodiff.Scalar(2.0)
y = x

x.grad = 0.0
y.grad = 1.0
y.backward()

print(x.grad)


# addition of scalars
z = x + x
x.grad = 0.0
z.grad = 1.0
z.backward()
print(x.grad)


# multiplacation
a = autodiff.Scalar(3.0)
b = a * a

a.grad = 0.0
b.grad = 1.0
b.backward()
print(a.grad)

# a bit more complex function

k = autodiff.Scalar(3.0)
l = (k * k * k) + (autodiff.Scalar(4.0) * k) + autodiff.Scalar(1.0)

k.grad = 0
l.grad = 1
l.backward()
print(k.grad)