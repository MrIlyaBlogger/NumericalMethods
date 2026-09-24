from pathlib import Path

EPS = 1e-12


def read_system(filename):
    data = Path(filename).read_text(encoding="utf-8").split()
    if not data:
        raise ValueError("Файл с входными данными пуст.")
    n = int(data[0])
    if n <= 0:
        raise ValueError("Размерность матрицы должна быть положительной.")
    expected = 1 + n * n + n
    if len(data) != expected:
        raise ValueError(f"Ожидалось {expected - 1} чисел после размерности, получено {len(data) - 1}.")
    values = list(map(float, data[1:]))
    return [values[i * n:(i + 1) * n] for i in range(n)], values[n * n:n * n + n]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def multiply_matrices(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def multiply_matrix_vector(a, b):
    return [sum(a[i][j] * b[j] for j in range(len(b))) for i in range(len(a))]


def lu_decomposition(a):
    n = len(a)
    u, l, p = [row[:] for row in a], identity(n), identity(n)
    swaps = 0
    for k in range(n - 1):
        pivot_row = max(range(k, n), key=lambda row: abs(u[row][k]))
        if abs(u[pivot_row][k]) < EPS:
            raise ValueError("Матрица вырожденная.")
        if pivot_row != k:
            u[k], u[pivot_row] = u[pivot_row], u[k]
            p[k], p[pivot_row] = p[pivot_row], p[k]
            if k > 0:
                l[k][:k], l[pivot_row][:k] = l[pivot_row][:k], l[k][:k]
            swaps += 1
        pivot = u[k][k]
        for i in range(k + 1, n):
            factor = u[i][k] / pivot
            l[i][k] = factor
            for j in range(k, n):
                u[i][j] -= factor * u[k][j]
            u[i][k] = 0.0
    if abs(u[n - 1][n - 1]) < EPS:
        raise ValueError("Матрица вырожденная.")
    return l, u, p, swaps


def forward_substitution(l, b):
    n, y = len(l), [0.0] * len(l)
    for i in range(n):
        value = b[i] - sum(l[i][j] * y[j] for j in range(i))
        if abs(l[i][i]) < EPS:
            raise ValueError("Нулевой диагональный элемент матрицы L.")
        y[i] = value / l[i][i]
    return y


def back_substitution(u, b):
    n, x = len(u), [0.0] * len(u)
    for i in range(n - 1, -1, -1):
        value = b[i] - sum(u[i][j] * x[j] for j in range(i + 1, n))
        if abs(u[i][i]) < EPS:
            raise ValueError("Нулевой диагональный элемент матрицы U.")
        x[i] = value / u[i][i]
    return x


def solve_lu(l, u, p, b):
    return back_substitution(u, forward_substitution(l, multiply_matrix_vector(p, b)))


def inverse_lu(l, u, p):
    n = len(l)
    cols = [solve_lu(l, u, p, [1.0 if i == col else 0.0 for i in range(n)]) for col in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def determinant_lu(u, swaps):
    det = -1.0 if swaps % 2 else 1.0
    for i in range(len(u)):
        det *= u[i][i]
    return det


def print_matrix(name, matrix):
    print(name)
    for row in matrix:
        print(" ".join(f"{v:12.6f}" for v in row))
    print()


def print_vector(name, vector):
    print(name)
    print(" ".join(f"{v:12.6f}" for v in vector))
    print()


def max_matrix_error(a: list[list[float]], b: list[list[float]]) -> float:
    return max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0])))


def max_vector_error(a: list[float], b: list[float]) -> float:
    return max(abs(a[i] - b[i]) for i in range(len(a)))


def main():
    a, b = read_system("input1.1.txt")
    l, u, p, swaps = lu_decomposition(a)
    x = solve_lu(l, u, p, b)
    inverse = inverse_lu(l, u, p)
    e = identity(len(a))
    lu_product = multiply_matrices(l, u)
    pa_product = multiply_matrices(p, a)
    inv_check = multiply_matrices(a, inverse)
    sol_check = multiply_matrix_vector(a, x)

    print_matrix("Исходная матрица A:", a)
    print_vector("Вектор b:", b)
    print_matrix("Матрица L:", l)
    print_matrix("Матрица U:", u)
    print_matrix("Матрица перестановок P:", p)
    print_matrix("Результат L * U:", lu_product)
    print_matrix("Результат P * A:", pa_product)
    print_vector("Решение системы Ax = b:", x)
    print_matrix("Обратная матрица A^(-1), найденная через LU-разложение:", inverse)
    print(f"Определитель det(A) = {determinant_lu(u, swaps):.12f}")
    print(f"Количество перестановок строк: {swaps}\n")
    print_matrix("Проверка A * A^(-1):", inv_check)
    print_matrix("Единичная матрица E:", e)
    print(f"Максимальная ошибка A * A^(-1) - E: {max_matrix_error(inv_check, e):.3e}")
    print(f"Максимальная ошибка L * U - P * A: {max_matrix_error(lu_product, pa_product):.3e}")
    print_vector("Проверка A * x:", sol_check)
    print_vector("Исходный вектор b:", b)
    print(f"Максимальная ошибка A * x - b: {max_vector_error(sol_check, b):.3e}")


if __name__ == "__main__":
    main()