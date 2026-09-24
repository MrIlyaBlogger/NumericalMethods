from pathlib import Path
import math

EPS = 1e-10
MAX_ITER = 1000


def read_matrix(filename: str) -> list[list[float]]:
    data = Path(filename).read_text(encoding="utf-8").split()

    if not data:
        raise ValueError("Файл с входными данными пуст.")

    n = int(data[0])

    if n <= 0:
        raise ValueError("Размерность матрицы должна быть положительной.")

    expected = 1 + n * n

    if len(data) != expected:
        raise ValueError(
            f"Ожидалось {expected - 1} чисел после размерности, "
            f"получено {len(data) - 1}."
        )

    values = list(map(float, data[1:]))
    matrix = []
    index = 0

    for _ in range(n):
        matrix.append(values[index:index + n])
        index += n

    return matrix


def print_matrix(name: str, matrix: list[list[float]]) -> None:
    print(name)
    for row in matrix:
        print(" ".join(f"{value:12.6f}" for value in row))
    print()


def print_vector(name: str, vector: list[float]) -> None:
    print(name)
    print(" ".join(f"{value:12.6f}" for value in vector))
    print()


def multiply_matrices(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
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


def jacobi_eigen(
    a: list[list[float]],
    eps: float = EPS,
    max_iter: int = MAX_ITER
) -> tuple[list[float], list[list[float]], int]:
    n = len(a)
    a = [row[:] for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    iterations = 0

    for _ in range(max_iter):
        max_val = 0.0
        p = 0
        q = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > max_val:
                    max_val = abs(a[i][j])
                    p = i
                    q = j

        if max_val < eps:
            break

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        tau = (aqq - app) / (2.0 * apq)

        if tau >= 0:
            t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))

        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for i in range(n):
            if i != p and i != q:
                aip = a[i][p]
                aiq = a[i][q]

                a[i][p] = c * aip - s * aiq
                a[p][i] = a[i][p]

                a[i][q] = s * aip + c * aiq
                a[q][i] = a[i][q]

        for i in range(n):
            vip = v[i][p]
            viq = v[i][q]

            v[i][p] = c * vip - s * viq
            v[i][q] = s * vip + c * viq

        iterations += 1
    else:
        print("Предупреждение: достигнуто максимальное число итераций.")

    eigenvalues = [a[i][i] for i in range(n)]
    return eigenvalues, v, iterations


def check_eigen(
    a: list[list[float]],
    v: list[list[float]],
    eigenvalues: list[float]
) -> tuple[float, list[list[float]], list[list[float]]]:
    n = len(a)
    av = multiply_matrices(a, v)
    vd = [[v[i][j] * eigenvalues[j] for j in range(n)] for i in range(n)]
    max_err = max(
        abs(av[i][j] - vd[i][j])
        for i in range(n)
        for j in range(n)
    )
    return max_err, av, vd


def main() -> None:
    try:
        a = read_matrix("input1.4.txt")

        print("=== Метод вращений Якоби ===")
        print(f"Заданная точность: {EPS:.1e}\n")

        print_matrix("Исходная матрица A:", a)

        eigenvalues, eigenvectors, iterations = jacobi_eigen(a, EPS)

        print_vector("Найденные собственные значения:", eigenvalues)
        print(f"Количество итераций: {iterations}\n")

        print_matrix("Матрица собственных векторов V (по столбцам):", eigenvectors)

        max_err, av, vd = check_eigen(a, eigenvectors, eigenvalues)

        print_matrix("Проверка A * V:", av)
        print_matrix("Проверка V * Λ:", vd)
        print(f"Максимальная ошибка A*V - V*Λ: {max_err:.3e}")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()