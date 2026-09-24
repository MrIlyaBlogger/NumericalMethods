from pathlib import Path

EPS = 1e-6
MAX_ITER = 10000


def read_system(filename: str) -> tuple[list[list[float]], list[float]]:
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

    for i in range(n):
        a.append(values[index:index + n])
        index += n

    b = values[index:index + n]

    return a, b


def print_matrix(name: str, matrix: list[list[float]]) -> None:
    print(name)
    for row in matrix:
        print(" ".join(f"{value:12.6f}" for value in row))
    print()


def print_vector(name: str, vector: list[float]) -> None:
    print(name)
    print(" ".join(f"{value:12.6f}" for value in vector))
    print()


def simple_iteration(
        a: list[list[float]],
        b: list[float],
        eps: float = EPS,
        max_iter: int = MAX_ITER
) -> tuple[list[float], int, bool]:

    n = len(a)
    x = [0.0] * n
    x_new = [0.0] * n
    iterations = 0

    for k in range(max_iter):
        for i in range(n):
            s = sum(a[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / a[i][i]

        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        x = x_new[:]
        iterations += 1

        if diff < eps:
            return x, iterations, True

    return x, iterations, False


def seidel_method(
        a: list[list[float]],
        b: list[float],
        eps: float = EPS,
        max_iter: int = MAX_ITER
) -> tuple[list[float], int, bool]:

    n = len(a)
    x = [0.0] * n
    iterations = 0

    for k in range(max_iter):
        x_old = x[:]
        for i in range(n):
            s1 = sum(a[i][j] * x[j] for j in range(i))
            s2 = sum(a[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / a[i][i]

        diff = max(abs(x[i] - x_old[i]) for i in range(n))
        iterations += 1

        if diff < eps:
            return x, iterations, True

    return x, iterations, False


def check_solution(
        a: list[list[float]],
        b: list[float],
        x: list[float]
) -> list[float]:

    n = len(a)
    residual = [0.0] * n
    for i in range(n):
        s = sum(a[i][j] * x[j] for j in range(n))
        residual[i] = s - b[i]
    return residual


def get_reference_solution(
        a: list[list[float]],
        b: list[float]
) -> list[float] | None:

    try:
        import numpy as np
        return np.linalg.solve(np.array(a), np.array(b)).tolist()
    except ImportError:
        print("NumPy не установлен. Эталонное решение не может быть найдено.")
        return None


def main() -> None:
    try:
        a, b = read_system("input1.3.txt")

        print(f"=== Метод простых итераций и метод Зейделя ===")
        print(f"Заданная точность: {EPS:.1e}\n")

        print_matrix("Исходная матрица A:", a)
        print_vector("Вектор b:", b)

        reference_x = get_reference_solution(a, b)
        if reference_x:
            print_vector("Эталонное решение (через NumPy):", reference_x)

        print("--- Метод простых итераций ---")
        x_jacobi, iter_jacobi, converged_jacobi = simple_iteration(a, b, EPS)

        if converged_jacobi:
            print_vector("Решение:", x_jacobi)
            print(f"Количество итераций: {iter_jacobi}")
            residual_jacobi = check_solution(a, b, x_jacobi)
            max_res_jacobi = max(abs(r) for r in residual_jacobi)
            print(f"Максимальная невязка: {max_res_jacobi:.3e}")
        else:
            print("Метод не сошелся за указанное число итераций.")
        print()

        print("--- Метод Зейделя ---")
        x_seidel, iter_seidel, converged_seidel = seidel_method(a, b, EPS)

        if converged_seidel:
            print_vector("Решение:", x_seidel)
            print(f"Количество итераций: {iter_seidel}")
            residual_seidel = check_solution(a, b, x_seidel)
            max_res_seidel = max(abs(r) for r in residual_seidel)
            print(f"Максимальная невязка: {max_res_seidel:.3e}")
        else:
            print("Метод не сошелся за указанное число итераций.")
        print()

        print("=== Сравнение методов ===")
        if converged_jacobi and converged_seidel:
            print(f"Метод простых итераций: {iter_jacobi} итераций")
            print(f"Метод Зейделя:          {iter_seidel} итераций")

            if iter_seidel < iter_jacobi:
                speedup = iter_jacobi / iter_seidel
                print(f"Метод Зейделя сошелся быстрее в {speedup:.2f} раз(а).")
            elif iter_jacobi < iter_seidel:
                print("Метод простых итераций сошелся быстрее.")
            else:
                print("Оба метода сошлись за одинаковое число итераций.")

            if reference_x is not None:
                diff_jacobi = max(abs(x_jacobi[i] - reference_x[i]) for i in range(len(reference_x)))
                diff_seidel = max(abs(x_seidel[i] - reference_x[i]) for i in range(len(reference_x)))

                print(f"\nОтклонение от эталона (метод простых итераций): {float(diff_jacobi):.3e}")
                print(f"Отклонение от эталона (метод Зейделя):          {float(diff_seidel):.3e}")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()