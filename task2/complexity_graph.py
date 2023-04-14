import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    x: np.ndarray = np.linspace(0, 1024, 10240)
    y_257: np.ndarray = convolution_complexity(x, 257)
    y_129: np.ndarray = convolution_complexity(x, 129)
    y_65: np.ndarray = convolution_complexity(x, 65)
    y_33: np.ndarray = convolution_complexity(x, 33)
    plt.style.use("fast")
    plt.plot(x, y_257, label="257")
    plt.plot(x, y_129, label="129")
    plt.plot(x, y_65, label="65")
    plt.plot(x, y_33, label="33")
    plt.legend(loc="upper left", title="Template Side Length (k)")
    plt.xlabel("Image Side Length (N)")
    plt.ylabel("Number of Operations")
    plt.show()


def convolution_complexity(x: np.ndarray, k: int) -> np.ndarray:
    return (x**2) * (k**2)


if __name__ == "__main__":
    main()
