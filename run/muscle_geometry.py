import numpy as np

# import casadi 

# a = casadi.MX.sym("a", 3, 2)
# b = casadi.MX.sym("b", 2, 6)
# c = casadi.MX.sym("c", 3, 2)

# d = (c + a) @ (b * b)

# d_func = casadi.Function("d", [a, b, c], [d]).expand()
# d_jaco_func = casadi.Function("d", [a, b, c], [casadi.jacobian(d, a)])
# print(d_func(np.ones((3, 2)), np.ones((2, 6)), np.ones((3, 2))))
# print(d_jaco_func(np.ones((3, 2)), np.ones((2, 6)), np.ones((3, 2))))

def main():
    origin_point = np.array([0, 0, 0])
    insertion_point = np.array([0, 0, 1])


if __name__ == "__main__":
    main()
