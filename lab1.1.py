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
        raise ValueError(
            f"Ожидалось {expected - 1} чисел после размерности, "
            f"получено {len(data) - 1}."
        )

    values = list(map(float, data[1:]))

    a = []
    index = 0

    for _ in range(n):
        a.append(values[index:index + n])
        index += n

    b = values[index:index + n]

    return a, b


def copy_matrix(matrix):
    return [row[:] for row in matrix]


def identity_matrix(n):
    return [
        [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]


def multiply_matrices(a, b):
    rows = len(a)
    cols = len(b[0])
    inner = len(b)

    return [
        [
            sum(a[i][k] * b[k][j] for k in range(inner))
            for j in range(cols)
        ]
        for i in range(rows)
    ]


def multiply_matrix_vector(a, b):
    return [
        sum(a[i][j] * b[j] for j in range(len(b)))
        for i in range(len(a))
    ]


def lu_decomposition(a):
    n = len(a)
    u = copy_matrix(a)
    l = identity_matrix(n)
    p = identity_matrix(n)
    swaps = 0

    for k in range(n - 1):
        pivot_row = max(
            range(k, n),
            key=lambda row: abs(u[row][k])
        )

        if abs(u[pivot_row][k]) < EPS:
            raise ValueError("Матрица вырожденная.")

        if pivot_row != k:
            u[k], u[pivot_row] = (
                u[pivot_row],
                u[k],
            )

            p[k], p[pivot_row] = (
                p[pivot_row],
                p[k],
            )

            if k > 0:
                l[k][:k], l[pivot_row][:k] = (
                    l[pivot_row][:k],
                    l[k][:k],
                )

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
    n = len(l)
    y = [0.0] * n

    for i in range(n):
        value = b[i]

        for j in range(i):
            value -= l[i][j] * y[j]

        if abs(l[i][i]) < EPS:
            raise ValueError(
                "Нулевой диагональный элемент матрицы L."
            )

        y[i] = value / l[i][i]

    return y


def back_substitution(u, b):
    n = len(u)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        value = b[i]

        for j in range(i + 1, n):
            value -= u[i][j] * x[j]

        if abs(u[i][i]) < EPS:
            raise ValueError(
                "Нулевой диагональный элемент матрицы U."
            )

        x[i] = value / u[i][i]

    return x


def solve_lu(l, u, p, b):
    permuted_b = multiply_matrix_vector(p, b)
    y = forward_substitution(l, permuted_b)
    x = back_substitution(u, y)

    return x


def inverse_lu(l, u, p):
    n = len(l)
    inverse = [[0.0] * n for _ in range(n)]

    for column in range(n):
        e = [0.0] * n
        e[column] = 1.0

        current_column = solve_lu(l, u, p, e)

        for i in range(n):
            inverse[i][column] = current_column[i]

    return inverse


def determinant_lu(u, swaps):
    determinant = -1.0 if swaps % 2 else 1.0

    for i in range(len(u)):
        determinant *= u[i][i]

    return determinant


def print_matrix(name, matrix):
    print(name)

    for row in matrix:
        print(" ".join(f"{value:12.6f}" for value in row))

    print()


def print_vector(name, vector):
    print(name)
    print(" ".join(f"{value:12.6f}" for value in vector))
    print()


def max_matrix_error(a: list[list[float]], b: list[list[float]]) -> float:
    return max(
        abs(a[i][j] - b[i][j])
        for i in range(len(a))
        for j in range(len(a[0]))
    )


def max_vector_error(a: list[float], b: list[float]) -> float:
    return max(
        abs(a[i] - b[i])
        for i in range(len(a))
    )


def main():
    a, b = read_system("input1.1.txt")

    l, u, p, swaps = lu_decomposition(a)

    x = solve_lu(l, u, p, b)
    inverse = inverse_lu(l, u, p)
    determinant = determinant_lu(u, swaps)

    identity = identity_matrix(len(a))

    lu_product = multiply_matrices(l, u)
    pa_product = multiply_matrices(p, a)
    inverse_check = multiply_matrices(a, inverse)
    solution_check = multiply_matrix_vector(a, x)

    print_matrix("Исходная матрица A:", a)
    print_vector("Вектор b:", b)

    print_matrix("Матрица L:", l)
    print_matrix("Матрица U:", u)
    print_matrix("Матрица перестановок P:", p)

    print_matrix("Результат L * U:", lu_product)
    print_matrix("Результат P * A:", pa_product)

    print_vector("Решение системы Ax = b:", x)

    print_matrix(
        "Обратная матрица A^(-1), найденная через LU-разложение:",
        inverse
    )

    print(f"Определитель det(A) = {determinant:.12f}")
    print(f"Количество перестановок строк: {swaps}")
    print()

    print_matrix(
        "Проверка A * A^(-1):",
        inverse_check
    )

    print_matrix(
        "Единичная матрица E:",
        identity
    )

    print(
        "Максимальная ошибка A * A^(-1) - E: "
        f"{max_matrix_error(inverse_check, identity):.3e}"
    )

    print(
        "Максимальная ошибка L * U - P * A: "
        f"{max_matrix_error(lu_product, pa_product):.3e}"
    )

    print_vector("Проверка A * x:", solution_check)
    print_vector("Исходный вектор b:", b)

    print(
        "Максимальная ошибка A * x - b: "
        f"{max_vector_error(solution_check, b):.3e}"
    )


if __name__ == "__main__":
    main()