import abc
import itertools as it
import json
import random
from collections import defaultdict

import numpy as np
import scipy.stats
from tqdm import tqdm


class Predicate(abc.ABC):
    @abc.abstractmethod
    def satisfies(self, word: str) -> bool:
        raise NotImplemented

    @abc.abstractmethod
    def emoji(self, dark_mode=True):
        raise NotImplemented


class GreenMatch(Predicate):
    def __init__(self, idx, letter):
        self.idx = idx
        self.letter = letter

    def satisfies(self, word: str) -> bool:
        return word[self.idx] == self.letter

    def emoji(self, dark_mode=True):
        return "🟩"


class YellowMatch(Predicate):
    def __init__(self, idx, letter):
        self.idx = idx
        self.letter = letter

    def satisfies(self, word: str) -> bool:
        return self.letter in word and word[self.idx] != self.letter

    def emoji(self, dark_mode=True):
        return "🟨"


class BlackMatch(Predicate):
    def __init__(self, letter):
        self.letter = letter

    def satisfies(self, word: str) -> bool:
        return self.letter not in word

    def emoji(self, dark_mode=True):
        return "⬛" if dark_mode else "⬜"


def make_predicates(guess: str, solution: str) -> [Predicate]:
    preds = []
    for idx, (guess_letter, solution_letter) in enumerate(zip(guess, solution)):
        if guess_letter == solution_letter:
            pred = GreenMatch(idx, guess_letter)
        elif guess_letter in solution:
            pred = YellowMatch(idx, guess_letter)
        else:
            pred = BlackMatch(guess_letter)
        preds.append(pred)
    return preds


def make_response_codes(guess: str, solution: str) -> np.ndarray:
    preds = np.zeros(5, dtype=np.uint8)
    for idx, (guess_letter, solution_letter) in enumerate(zip(guess, solution)):
        if guess_letter == solution_letter:
            pred = 2
        elif guess_letter in solution:
            pred = 1
        else:
            pred = 0
        preds[idx] = pred
    # convert base-3 representation to int
    return np.sum(preds * 3**np.arange(5), dtype=np.uint8)


def convert_code(code: np.uint8, guess: str) -> [Predicate]:
    code = (code // 3**np.arange(5)) % 3  # convert int to base-3 representation
    preds = []
    for idx, (c, letter) in enumerate(zip(code, guess)):
        if c == 2:
            pred = GreenMatch(idx, letter)
        elif c == 1:
            pred = YellowMatch(idx, letter)
        elif c == 0:
            pred = BlackMatch(letter)
        else:
            raise ValueError()
        preds.append(pred)
    return preds


class Policy(abc.ABC):
    @abc.abstractmethod
    def observe(self, guess: str, predicates: [Predicate]):
        raise NotImplemented

    @abc.abstractmethod
    def guess(self) -> str:
        raise NotImplemented


class RandomValidPolicy(Policy):
    def __init__(self, list_='solutions'):
        self.predicates = []
        self.list_ = list_

    def observe(self, guess: str, predicates: [Predicate]):
        self.predicates += predicates

    def guess(self) -> str:
        valid_guesses = [w for w in word_lists[self.list_] if all(p.satisfies(w) for p in self.predicates)]
        return random.choice(valid_guesses)


class GreedyHeuristicPolicy(Policy):
    def __init__(self, statistic=np.mean, n_attempts=6):
        self.predicates = []
        self.statistic = statistic
        self.remaining = n_attempts

    def observe(self, guess: str, predicates: [Predicate]):
        self.predicates += predicates
        self.remaining -= 1

    def guess(self) -> str:
        guess_list = word_lists['guesses']
        solution_list = word_lists['solutions']
        possible = np.array([all(p.satisfies(w) for p in self.predicates) for w in solution_list], dtype=np.bool_)
        if self.remaining >= possible.sum():  # limit guesses to valid solutions
            solutions = [s for (p, s) in zip(possible, solution_list) if p]
            guess_idcs = [i for i, g in enumerate(guess_list) if g in solutions]
            possible_clue_matrix = clue_matrix[guess_idcs][:, possible]
            guess_list = [guess_list[i] for i in guess_idcs]
        else:
            possible_clue_matrix = clue_matrix[:, possible]
        # pcm_decimal = np.sum([possible_clue_matrix[:, :, i] * 3**i for i in range(5)], axis=0)
        clue_counts = [np.unique(row, return_counts=True)[1] for row in possible_clue_matrix]
        entropies = [scipy.stats.entropy(row) for row in clue_counts]
        action_idx = np.argmax(entropies)
        return guess_list[action_idx]


# Too inefficient; probably broken too
class GreedyPolicy(Policy):
    def __init__(self, statistic=np.mean):
        self.predicates = []
        self.statistic = statistic

    def observe(self, guess: str, predicates: [Predicate]):
        self.predicates += predicates

    def guess(self) -> str:
        guess_list = word_lists['guesses']
        solution_list = word_lists['solutions']
        # possible_solutions = [w for w in word_lists['solutions'] if all(p.satisfies(w) for p in self.predicates)]
        possible = [all(p.satisfies(w) for p in self.predicates) for w in solution_list]
        remaining_matrix = np.zeros_like(clue_matrix, dtype=np.uint64)
        for i, j in it.product(range(len(guess_list)), range(len(solution_list))):
            # clue = clue_matrix[guess_list[i]][possible_solutions[j]]
            clue_code = clue_matrix[i, j]
            guess = guess_list[i]
            clue = convert_code(clue_code, guess)
            remaining_words = [all(p.satisfies(w) for p in clue) and possible[wi] for (wi, w) in enumerate(solution_list)]
            remaining_matrix[i, j] = len(remaining_words)
        remaining_list = self.statistic(remaining_matrix, axis=1)
        action_idx = remaining_list.argmin()
        return guess_list[action_idx]


class InteractivePolicy(Policy):
    def guess(self) -> str:
        s = input()
        while s not in word_lists['guesses']:
            print("Not in word list. Try again")
            s = input()
        return s

    def observe(self, guess: str, predicates: [Predicate]):
        s = "".join(p.emoji() for p in predicates)
        print(s)


class Game:
    def __init__(self, solution=None, n_attempts=6, policy=None):
        self.solution = solution or random.choice(word_lists['solutions'])
        self.policy = policy or RandomValidPolicy()
        self.n_attempts = n_attempts

    def play(self):
        guesses = []
        clues = []
        for t in range(self.n_attempts):
            guess = self.policy.guess()
            clue = make_predicates(guess, self.solution)
            self.policy.observe(guess, clue)
            guesses.append(guess)
            clues.append(clue)
            if guess == self.solution:
                break
        return guesses, clues


print("Loading word lists")
with open('words.json', 'rb') as fp:
    word_lists = json.load(fp)
    word_lists['guesses'] += word_lists['solutions']
    word_lists['solutions'] = [w for w in word_lists['solutions'] if all(w[i] not in w[i+1:] for i in range(4))]

try:
    clue_matrix = np.load('clue_matrix.npy')
except IOError:
    print("Computing clue matrix")
    clue_matrix = np.zeros((len(word_lists['guesses']), len(word_lists['solutions'])), dtype=np.uint8)
    iterator = it.product(enumerate(word_lists['guesses']), enumerate(word_lists['solutions']))
    for (i, a), (j, w) in tqdm(iterator, total=len(word_lists['guesses']) * len(word_lists['solutions'])):
        clue_matrix[i, j] = make_response_codes(a, w)
    np.save('clue_matrix.npy', clue_matrix)
    print("Made clue matrix")


def main():
    results = {'statistics': defaultdict(int), 'games': []}
    for r, solution in tqdm(enumerate(word_lists['solutions']), total=len(word_lists['solutions'])):
        game = Game(policy=GreedyHeuristicPolicy(), solution=solution)
        guesses, clues = game.play()
        results['games'].append({'solution': solution, 'guesses': guesses})
        if guesses[-1] == game.solution:
            results['statistics'][len(guesses)] += 1
        else:
            results['statistics'][-1] += 1
    with open('results.json', mode='w') as fp:
        json.dump(results, fp)
    print(results['statistics'])

    # clue_strs = ["".join(p.emoji() for p in preds) for preds in clues]
    # for guess, clue_str in zip(guesses, clue_strs):
    #     print(f"{guess}\t{clue_str}")


if __name__ == '__main__':
    main()
