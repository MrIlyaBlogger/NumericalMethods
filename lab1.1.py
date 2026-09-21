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


def gaussian_elimination(a, b):
    n = len(a)
    matrix = copy_matrix(a)
    vector = b[:]
    swaps = 0

    for k in range(n - 1):
        pivot_row = max(
            range(k, n),
            key=lambda i: abs(matrix[i][k])
        )

        if abs(matrix[pivot_row][k]) < EPS:
            raise ValueError(
                "Система не имеет единственного решения."
            )

        if pivot_row != k:
            matrix[k], matrix[pivot_row] = (
                matrix[pivot_row],
                matrix[k],
            )
            vector[k], vector[pivot_row] = (
                vector[pivot_row],
                vector[k],
            )
            swaps += 1

        for i in range(k + 1, n):
            factor = matrix[i][k] / matrix[k][k]

            for j in range(k, n):
                matrix[i][j] -= factor * matrix[k][j]

            vector[i] -= factor * vector[k]
            matrix[i][k] = 0.0

    if abs(matrix[n - 1][n - 1]) < EPS:
        raise ValueError(
            "Система не имеет единственного решения."
        )

    return matrix, vector, swaps


def back_substitution(a, b):
    n = len(a)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        value = b[i]

        for j in range(i + 1, n):
            value -= a[i][j] * x[j]

        if abs(a[i][i]) < EPS:
            raise ValueError(
                "Нулевой диагональный элемент."
            )

        x[i] = value / a[i][i]

    return x


def solve_gauss(a, b):
    triangular, new_b, swaps = gaussian_elimination(a, b)
    x = back_substitution(triangular, new_b)

    return x, triangular, new_b, swaps


def determinant_gauss(a):
    triangular, _, swaps = gaussian_elimination(
        a,
        [0.0] * len(a)
    )

    determinant = -1.0 if swaps % 2 else 1.0

    for i in range(len(a)):
        determinant *= triangular[i][i]

    return determinant


def inverse_gauss(a):
    n = len(a)
    augmented = [
        a[i][:] + identity_matrix(n)[i]
        for i in range(n)
    ]

    swaps = 0

    for k in range(n):
        pivot_row = max(
            range(k, n),
            key=lambda i: abs(augmented[i][k])
        )

        if abs(augmented[pivot_row][k]) < EPS:
            raise ValueError("Матрица вырожденная.")

        if pivot_row != k:
            augmented[k], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[k],
            )
            swaps += 1

        pivot = augmented[k][k]

        for j in range(2 * n):
            augmented[k][j] /= pivot

        for i in range(n):
            if i == k:
                continue

            factor = augmented[i][k]

            for j in range(2 * n):
                augmented[i][j] -= factor * augmented[k][j]

            augmented[i][k] = 0.0

    return [row[n:] for row in augmented]


def print_matrix(name, matrix):
    print(name)

    for row in matrix:
        print(" ".join(f"{value:12.6f}" for value in row))

    print()


def print_vector(name, vector):
    print(name)
    print(" ".join(f"{value:12.6f}" for value in vector))
    print()


def max_matrix_error(a, b):
    return max(
        abs(a[i][j] - b[i][j])
        for i in range(len(a))
        for j in range(len(a[0]))
    )


def max_vector_error(a, b):
    return max(
        abs(a[i] - b[i])
        for i in range(len(a))
    )


def main():
    a, b = read_system("input.txt")

    x, triangular, transformed_b, swaps = solve_gauss(a, b)
    inverse = inverse_gauss(a)
    determinant = determinant_gauss(a)

    identity = identity_matrix(len(a))
    inverse_check = multiply_matrices(a, inverse)
    solution_check = multiply_matrix_vector(a, x)

    print_matrix("Исходная матрица A:", a)
    print_vector("Вектор b:", b)

    print_matrix(
        "Матрица после прямого хода метода Гаусса:",
        triangular
    )
    print_vector(
        "Правая часть после прямого хода:",
        transformed_b
    )

    print_vector(
        "Решение системы Ax = b:",
        x
    )

    print_matrix(
        "Обратная матрица A^(-1):",
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

    print_vector(
        "Проверка A * x:", solution_check)

    print_vector(
        "Исходный вектор b:", b)

    print(
        "Максимальная ошибка A * x - b: " f"{max_vector_error(solution_check, b):.3e}")


if __name__ == "__main__":
    main()
