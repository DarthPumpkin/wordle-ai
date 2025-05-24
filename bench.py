from datetime import datetime

import numba
import numpy as np


max_clues = 3 ** 5


@numba.njit
def clue_entropies(clue_matrix: np.ndarray, probs: np.ndarray) -> np.ndarray:
    rows, cols = clue_matrix.shape
    entropies = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        weighted_counts = np.zeros(max_clues, dtype=np.float64)
        for j in range(cols):
            weighted_counts[clue_matrix[i, j]] += probs[j]
        entropies[i] = entropy_bits(weighted_counts)

    return entropies


@numba.njit
def entropy_bits(vec: np.ndarray) -> np.ndarray:
    mask = vec > 0
    return np.sum(-vec[mask] * np.log2(vec[mask]))


def bench(rows, cols):
    clue_matrix = np.random.randint(low=0, high=3 ** 5, size=(rows, cols))
    clue_matrix = np.astype(clue_matrix, np.uint8)
    probs = np.ones(cols, dtype=np.float64) / cols

    # Prime the JIT with an array of the same shape
    clue_entropies(clue_matrix, np.random.uniform(size=cols))

    tic = datetime.now()
    clue_entropies(clue_matrix, probs)
    toc = datetime.now()
    print(f"{rows} x {cols}:")
    print((toc - tic))


def main():
    clue_matrix = [
        [1, 3, 4, 4],
        [0, 4, 0, 0],
        [0, 1, 2, 3],
        [5, 5, 5, 5],
    ]
    clue_matrix = np.asarray(clue_matrix)
    probs = np.ones(4, dtype=np.float64) / 4.
    print("Test")
    print(clue_entropies(clue_matrix, probs))

    rows = 1000
    cols = 100
    bench(rows, cols)

    rows = 100
    cols = 1000
    bench(rows, cols)

    rows = 1_000
    cols = 10_000
    bench(rows, cols)

    rows = 10_000
    cols = 1000
    bench(rows, cols)


if __name__ == '__main__':
    main()
