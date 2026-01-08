import numpy as np

def main():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    b_transposed = b.reshape(-1, 1)
    dot_product = np.dot(a, b)
    matrix_product = np.matmul(a.reshape(1, -1),b_transposed)

    print("Вектор a:", a)
    print("Вектор b:", b)
    print("Транспонированный вектор b (столбец):\n", b_transposed)
    print("Скалярное произведение a · b:", dot_product)
    print("Матричное произведение (a как строка) × (b как столбец):\n", matrix_product)

if __name__ == "__main__":
    main()