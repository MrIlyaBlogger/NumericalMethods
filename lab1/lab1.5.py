from pathlib import Path
import math
import importlib.util

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


def multiply_matrices(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]


def qr_decomposition(a: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    m = len(a)
    n = len(a[0])
    q = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    r = [row[:] for row in a]

    for k in range(min(m - 1, n)):
        x = [r[i][k] for i in range(k, m)]
        norm_x = math.sqrt(sum(v * v for v in x))
        if norm_x < 1e-15:
            continue

        alpha = -norm_x if x[0] > 0 else norm_x
        v = x[:]
        v[0] -= alpha
        norm_v = math.sqrt(sum(v_i * v_i for v_i in v))
        if norm_v < 1e-15:
            continue
        v = [v_i / norm_v for v_i in v]

        for j in range(k, n):
            dot = sum(v[i - k] * r[i][j] for i in range(k, m))
            for i in range(k, m):
                r[i][j] -= 2.0 * v[i - k] * dot

        for i in range(m):
            dot = sum(q[i][j] * v[j - k] for j in range(k, m))
            for j in range(k, m):
                q[i][j] -= 2.0 * dot * v[j - k]

    return q, r


def qr_algorithm(a: list[list[float]], eps: float = EPS, max_iter: int = MAX_ITER):
    n = len(a)
    a_k = [row[:] for row in a]

    for _ in range(max_iter):
        q, r = qr_decomposition(a_k)
        a_k = multiply_matrices(r, q)

        max_off = 0.0
        for i in range(1, n):
            for j in range(i):
                if abs(a_k[i][j]) > max_off:
                    max_off = abs(a_k[i][j])
        if max_off < eps:
            break

    eigenvalues = []
    i = 0
    while i < n:
        if i == n - 1 or abs(a_k[i + 1][i]) < eps:
            eigenvalues.append(a_k[i][i])
            i += 1
        else:
            a11 = a_k[i][i]
            a12 = a_k[i][i + 1]
            a21 = a_k[i + 1][i]
            a22 = a_k[i + 1][i + 1]
            trace = a11 + a22
            det = a11 * a22 - a12 * a21
            disc = trace * trace - 4 * det
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                lam1 = (trace + sqrt_disc) / 2
                lam2 = (trace - sqrt_disc) / 2
                eigenvalues.append(lam1)
                eigenvalues.append(lam2)
            else:
                sqrt_disc = math.sqrt(-disc)
                real_part = trace / 2
                imag_part = sqrt_disc / 2
                eigenvalues.append(complex(real_part, imag_part))
                eigenvalues.append(complex(real_part, -imag_part))
            i += 2

    return eigenvalues, a_k


def print_eigenvalues(eigenvalues: list) -> None:
    print("Собственные значения:")
    for val in eigenvalues:
        if isinstance(val, complex):
            print(f"  {val.real:12.6f} {val.imag:+12.6f}j")
        else:
            print(f"  {val:12.6f}")
    print()


def main() -> None:
    try:
        print("=" * 50)
        print("QR-разложение для матрицы")
        print("=" * 50)
        a = read_matrix("input1.5.txt")
        print_matrix("Исходная матрица A:", a)

        q, r = qr_decomposition(a)
        print_matrix("Матрица Q:", q)
        print_matrix("Матрица R:", r)

        qr_product = multiply_matrices(q, r)
        print_matrix("Результат Q * R:", qr_product)

        # Проверка Q*R = A
        max_err = max(abs(qr_product[i][j] - a[i][j])
                      for i in range(len(a)) for j in range(len(a[0])))
        print(f"Максимальная ошибка Q*R - A: {max_err:.3e}\n")

        # Собственные значения через QR-алгоритм
        eigenvalues, final_matrix = qr_algorithm(a)
        print_eigenvalues(eigenvalues)

        # --- Пример с комплексно-сопряжёнными собственными значениями ---
        print("=" * 50)
        print("Пример матрицы с комплексно-сопряжёнными собственными значениями")
        print("=" * 50)
        a_complex = read_matrix("input_complex.txt")
        print_matrix("Исходная матрица A:", a_complex)

        q2, r2 = qr_decomposition(a_complex)
        print_matrix("Матрица Q:", q2)
        print_matrix("Матрица R:", r2)
        qr_product2 = multiply_matrices(q2, r2)
        print_matrix("Результат Q * R:", qr_product2)

        eigenvalues2, _ = qr_algorithm(a_complex)
        print_eigenvalues(eigenvalues2)

        # Проверка с помощью NumPy (разрешено только для этого примера)
        if importlib.util.find_spec("numpy") is not None:
            import numpy as np
            np_eigs = np.linalg.eigvals(np.array(a_complex))
            print("Проверка с помощью NumPy:")
            for val in np_eigs:
                if isinstance(val, complex) or np.iscomplexobj(val):
                    print(f"  {val.real:12.6f} {val.imag:+12.6f}j")
                else:
                    print(f"  {val:12.6f}")
            print()
        else:
            print("NumPy не установлен, проверка комплексных значений пропущена.\n")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()