from pathlib import Path


EPS = 1e-12


def read_system(filename):
    data = Path(filename).read_text(encoding="utf-8").split()

    if not data:
        raise ValueError("Файл с входными данными пуст.")

    n = int(data[0])

    if n <= 0:
        raise ValueError("Размерность матрицы должна быть положительной.")

    expected = 1 + n + (n - 1) + (n - 1) + n

    if len(data) != expected:
        raise ValueError(
            f"Ожидалось {expected - 1} чисел после размерности, "
            f"получено {len(data) - 1}."
        )

    values = list(map(float, data[1:]))

    b = values[0:n]
    a = values[n:n + n - 1]
    c = values[n + n - 1:n + n - 1 + n - 1]
    d = values[n + n - 1 + n - 1:]

    return a, b, c, d


def progonka_algorithm(a, b, c, d):
    n = len(b)

    if len(a) != n - 1 or len(c) != n - 1 or len(d) != n:
        raise ValueError("Неверные размеры диагоналей.")
    alpha = [0.0] * (n - 1)
    beta = [0.0] * n
    x = [0.0] * n

    if abs(b[0]) < EPS:
        raise ValueError("Нулевой элемент на главной диагонали (b[0]).")

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denom = a[i - 1] * alpha[i - 1] + b[i]

        if abs(denom) < EPS:
            raise ValueError(f"Нулевой знаменатель на шаге {i}. Метод прогонки неприменим.")

        beta[i] = (d[i] - a[i - 1] * beta[i - 1]) / denom

        if i < n - 1:
            alpha[i] = -c[i] / denom

    x[n - 1] = beta[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return alpha, beta, x


def check_solution(a, b, c, d, x):
    n = len(b)
    residual = [0.0] * n

    residual[0] = b[0] * x[0] + c[0] * x[1] - d[0]

    for i in range(1, n - 1):
        residual[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1] - d[i]

    residual[n - 1] = a[n - 2] * x[n - 2] + b[n - 1] * x[n - 1] - d[n - 1]

    return residual


def print_vector(name, vector):
    print(name)
    print(" ".join(f"{value:12.6f}" for value in vector))
    print()


def main():
    try:
        a, b, c, d = read_system("input1.2.txt")

        print("=== Исходные данные ===")
        print_vector("Главная диагональ (b):", b)
        print_vector("Нижняя диагональ (a):", a)
        print_vector("Верхняя диагональ (c):", c)
        print_vector("Вектор правой части (d):", d)

        alpha, beta, x = progonka_algorithm(a, b, c, d)

        print("=== Прогоночные коэффициенты ===")
        print_vector("Коэффициенты alpha:", alpha)
        print_vector("Коэффициенты beta:", beta)

        print("=== Итоговое решение системы ===")
        print_vector("Вектор неизвестных x:", x)

        residual = check_solution(a, b, c, d, x)
        max_error = max(abs(r) for r in residual)

        print("=== Проверка решения ===")
        print_vector("Невязка (A*x - d):", residual)
        print(f"Максимальная невязка: {max_error:.3e}")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()