from matrix import Matrix, Vector

# test 42

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.shape)
# Output: (3, 2)

print(m1.T())
# Output: Matrix([[0., 2., 4.], [1., 3., 5.]])

print(m1.T().shape)
# Output: (2, 3)

m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.shape)
# Output: (2, 3)

print(m1.T())
# Output: Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

print(m1.T().shape)
# Output: (3, 2)

m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
print(m1 * m2)
# Output: Matrix([[28., 34.], [56., 68.]])

m1 = Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
v1 = Vector([[1], [2], [3]])
print(m1 * v1)
# Output: Matrix([[8], [16]])
# Or: Vector([[8], [16]

v1 = Vector([[1], [2], [3]])
v2 = Vector([[2], [4], [8]])
print(v1 + v2)
# Output: Vector([[3],[6],[11]])

# # Example 3:
v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
v2 = Vector([[2.0], [1.5], [2.25], [4.0]])
print("\nv1 = Vector([[0.0], [1.0], [2.0], [3.0]])\nv2 = Vector([[2.0], [1.5], [2.25], [4.0]])\nv1.dot(v2):\n", v1.dot(v2))
# # Expected output:
# # 18.0

v3 = Vector([[1.0, 3.0]])
v4 = Vector([[2.0, 4.0]])
print(
    "\nv3 = Vector([[1.0, 3.0]])\nv4 = Vector([[2.0, 4.0]])\nv3.dot(v4):\n", v3.dot(v4))
# Expected output:
# 14.0

list_01 = [[1, 2, 3], [4, 5, 6]]
shape_01 = (3, 2)
print(Matrix(list_01))
print(Matrix(shape_01))
print(Vector(list_01))
print(Vector(shape_01))

list_02 = [[1], [2], [3]]
shape_02 = (1, 3)
print(Vector(list_02))
print(Vector(shape_02))
print(Matrix())
print(Vector())

shape_03 = (1, 3)
print(Vector(shape_03).shape)

m1 = Matrix([[1, 2, 3], [3, 4, 5]])
m2 = Matrix([[1, 0, 1], [0, 1, 0]])
m3 = Matrix([[1, 0], [1, 1], [1, 0]])

print("\n--- Addition --- \n")

print(m1 + m2)

v1 = Vector([[1], [2], [3], [4], [5]])
v2 = Vector([[1], [0], [1], [1], [0]])

print(v1 + v2)

v3 = Vector([[1, 2, 3, 4, 5]])
v4 = Vector([[1, 0, 1, 1, 0]])
v5 = Vector([[1], [2], [3]])

print(v3 + v4)

print("\n--- Substraction --- \n")

print(m1 - m2)
print(m2 - m1)
print(v1 - v2)
print(v3 - v4)
print(v2 - v1)
print(v4 - v3)

print("\n--- Division by a scalar --- \n")

print(m1 / 2)
print(v1 / 2)
print(v3 / 2)
print(v3 / 0)

#  Error tests
print(v3 / v4)
print(3 / v4)

print("\n--- Multiplication --- \n")

print(m1 * m3)
print(m1 * 2)
print(m1 * v5)
print(v3 * v1)
print(v1 * v3)
print(v1 * 2)
print(v3 * 2)

#  Error tests
print(v3 * v4)

# # Example 6:
print("\nv5 = Vector([0.9, 1.0, 0.2, 3.0])")
v5 = Vector([[0.9, 1.0, 0.2, 3.0]])
print("v6 = Vector([0.4, 2.0, 0.5, 10])")
v6 = Vector([[0.4, 2.0, 0.5, 10]])
print("\nv5.dot(v6):\n", v5.dot(v6))

# # Example 7:
print("\nv5 = Vector([[0.9], [1.0], [0.2], [3.0]])")
v5 = Vector([[0.9], [1.0], [0.2], [3.0]])
print("v6 = Vector([[0.4], [2.0], [0.5], [10], [5]])\n")
v6 = Vector([[0.4], [2.0], [0.5], [10], [5]])
print("\nv5.dot(v6):\n", v5.dot(v6))
v6 = Vector([[0.4], [2.0], [0.5], [10]])
print("\nv5.dot(v6):\n", v5.dot(v6))
