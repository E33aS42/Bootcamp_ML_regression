class Matrix:
    def __init__(self, arg=[[]]):
        try:
            # get matrix data
            assert arg and isinstance(
                arg, (list, tuple)), "Matrix arguments must either be a list of lists or a tuple"

            # case: argument is a list
            if isinstance(arg, list):
                for v in arg:
                    assert isinstance(
                        v, list), "Matrix arguments must either be a list of lists or a tuple"
                    if v:
                        for i in v:
                            assert isinstance(
                                i, (int, float)), "Matrix elements must either be ints or floats"
                length = len(arg[0])
                for v in arg:
                    assert len(
                        v) == length, "Every matrix list elements must have the same length"
                self.data = arg
                self.shape = (len(arg), len(arg[0]))

            # case: argument is a tuple
            elif isinstance(arg, tuple):
                assert len(arg) == 2, "Tuple argument must be of size 2"
                assert isinstance(arg[0], int) and isinstance(
                    arg[1], int) and arg[0] > 0 and arg[1] > 0, "Tuple elements must be positive integers"
                self.data = [[0 for j in range(arg[1])] for i in range(arg[0])]
                self.shape = arg
        except AssertionError as msg:
            print(msg)
            self.data = None
            self.shape = None

    def __add__(self, other):
        try:
            assert isinstance(self, (Matrix, Vector)) and isinstance(
                other, (Matrix, Vector)), "both inputs must either be matrix or vector"
            assert self.data and other.data, "NotImplementedError"
            assert self.shape == other.shape, "Error: matrix have different shapes"
            res = [[0 for i in range(self.shape[1])]
                   for j in range(self.shape[0])]
            for j in range(self.shape[1]):
                for i in range(self.shape[0]):
                    res[i][j] = self.data[i][j] + other.data[i][j]
            if isinstance(self, Vector) and isinstance(other, Vector):
                return Vector(res)
            return Matrix(res)
        except AssertionError as msg:
            print(msg)

    def __radd__(self, other):
        return other.__add__(self)

    def __sub__(self, other):
        try:
            assert isinstance(self, (Matrix, Vector)) and isinstance(
                other, (Matrix, Vector)), "both inputs must either be matrix or vector"
            assert self.data and other.data, "NotImplementedError"
            assert self.shape == other.shape, "Error: matrix have different shapes"
            res = [[0 for i in range(self.shape[1])]
                   for j in range(self.shape[0])]
            for j in range(self.shape[1]):
                for i in range(self.shape[0]):
                    res[i][j] = self.data[i][j] - other.data[i][j]
            if isinstance(self, Vector) and isinstance(other, Vector):
                return Vector(res)
            return Matrix(res)
        except AssertionError as msg:
            print(msg)

    def __rsub__(self, other):
        return other.__sub__(self)

    def __truediv__(self, other):
        try:
            assert isinstance(self, (Matrix, Vector)) and isinstance(
                other, (int, float)), "NotImplementedError"
            assert other != 0, "ZeroDivisionError"
            res = [[0 for i in range(self.shape[1])]
                   for j in range(self.shape[0])]
            for j in range(self.shape[1]):
                for i in range(self.shape[0]):
                    res[i][j] = self.data[i][j] / other
            if isinstance(self, Vector):
                return Vector(res)
            return Matrix(res)

        except AssertionError as msg:
            print(msg)

    def __rtruediv__(self, other):
        try:
            raise NotImplementedError
        except NotImplementedError as msg:
            print(
                "NotImplementedError: Division of a scalar by a Vector or Matrix is not defined here.")

    def __mul__(self, other):
        try:
            # if self == other:
            # 	return self.dot(other)
            assert isinstance(self, (Matrix, Vector)) and isinstance(
                other, (Matrix, Vector, int, float)), "NotImplementedError"

            if isinstance(other, (int, float)):
                res = [[0 for i in range(self.shape[1])]
                       for j in range(self.shape[0])]
                for j in range(self.shape[1]):
                    for i in range(self.shape[0]):
                        res[i][j] = self.data[i][j] * other

            elif isinstance(other, (Matrix, Vector)):
                assert self.shape[1] == other.shape[0], "arguments shapes are not compatible"
                assert self.shape[0] >= other.shape[1], "arguments shapes are not compatible"
                res = [[0 for j in range(other.shape[1])]
                       for i in range(self.shape[0])]
                # if isinstance(other, (Vector, Matrix)):
                for j in range(other.shape[1]):
                    for i in range(self.shape[0]):
                        res[i][j] = sum([self.data[i][z] * other.data[z][j]
                                        for z in range(self.shape[1])])

            # res = [[0 for i in range(self.shape[1])] for j in range(self.shape[0])]
            # if  isinstance(other, (int, float)):
            # 	for j in range(self.shape[1]):
            # 		for i in range(self.shape[0]):
            # 			res[i][j] = self.data[i][j] * other
            # elif isinstance(other, Matrix):
            # 	for j in range(self.shape[1]):
            # 		for i in range(self.shape[0]):
            # 			res[i][j] = self.data[i][j] * other.data[i][j]
            # elif isinstance(other, Vector):
            # 	for j in range(self.shape[1]):
            # 		for i in range(self.shape[0]):
            # 			res[i][j] = self.data[i][j] * other.data[i][j]

            if isinstance(self, Vector) and (isinstance(other, (int, float))
                                             or (isinstance(other, Vector) and (len(res) == 1 or len(res[0]) == 1))):
                return Vector(res)
            return Matrix(res)
        except AssertionError as msg:
            print(msg)

    def __rmul__(self, other):
        return other.__mul__(self)

    def T(self):
        try:
            assert isinstance(self, (Matrix, Vector)
                              ), "input must be a matrix or vector"
            res = [[0 for j in range(self.shape[0])]
                   for i in range(self.shape[1])]
            for i in range(self.shape[1]):
                for j in range(self.shape[0]):
                    res[i][j] = self.data[j][i]
            if isinstance(self, Vector):
                return Vector(res)
            return Matrix(res)

        except AssertionError as msg:
            print(msg)

    def __str__(self):
        """ The __str__ method represents the class objects as a string.
        The __str__ method is called when the following functions are invoked on the object and return a string:
        print()
        str()
        If we have not defined the __str__, then it will call the __repr__ method. The __repr__ method returns a string that describes the pointer of the object by default (if the programmer does not define it)."""
        if self.data:
            return f'Matrix({self.data})'
        return str(self.data)

    def __repr__(self):
        if self.data:
            return f'Matrix({self.data})'
        return str(self.data)


class Vector(Matrix):
    def __init__(self, data=[[]]):
        """python has a super() function that will make the child class inherit all the methods and properties from its parent.
        """
        super().__init__(data)
        try:
            assert self.shape, "Vector is None"
            assert 1 in self.shape, "One of the shape elements must be 1"

        except AssertionError as msg:
            print(msg)
            self.data = None
            self.shape = None

    def dot(self, other):
        try:
            assert isinstance(self, Vector) and isinstance(
                other, Vector), "both inputs must be vectors"
            assert self.shape == other.shape, "Error: vectors have different shapes"
            if self.shape[0] >= self.shape[1]:
                for x, y in zip(self.data, other.data):
                    assert isinstance(x[0], (int, float)
                                      ), "NotImplementedError"
                    assert isinstance(y[0], (int, float)
                                      ), "NotImplementedError"
                    return sum([x[0] * y[0] for x, y in zip(self.data, other.data)])
            else:
                for x, y in zip(self.data[0], other.data[0]):
                    assert isinstance(x, (int, float)), "NotImplementedError"
                    assert isinstance(y, (int, float)), "NotImplementedError"
                    return sum([x * y for x, y in zip(self.data[0], other.data[0])])
        except AssertionError as msg:
            print(msg)

    def __str__(self):
        """ The __str__ method represents the class objects as a string.
        The __str__ method is called when the following functions are invoked on the object and return a string:
        print()
        str()
        If we have not defined the __str__, then it will call the __repr__ method. The __repr__ method returns a string that describes the pointer of the object by default (if the programmer does not define it)."""
        if self.data:
            return f'Vector({self.data})'
        return str(self.data)

    def __repr__(self):
        if self.data:
            return f'Vector({self.data})'
        return str(self.data)
