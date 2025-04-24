import abc
import argparse
import itertools as it
import json
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import scipy.stats
from tqdm import tqdm


Clue = list['Predicate']
WordLists = dict[str, list[str]]


class Predicate(abc.ABC):
    @abc.abstractmethod
    def satisfies(self, word: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def emoji(self, dark_mode=True):
        raise NotImplementedError


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


def make_predicates(guess: str, solution: str) -> list[Predicate]:
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


def make_response_code(guess: str, solution: str) -> np.uint8:
    """Convert a clue to an unsigned 8-bit integer.
    The clue is the response to a guess for a given solution.
    Each color in the clue is represented by a digit:
    - 0: black
    - 1: yellow
    - 2: green

    The 8-bit integer is then obtained by converting the clue to the five corresponding numbers
    and interpreting them as a base-3 number in Little Endian order.

    For example, the clue "🟩⬛🟨⬛⬛" would be represented as 00102 in base-3, which is
    2 * 3^0 + 0 * 3^1 + 1 * 3^2 + 0 * 3^3 + 0 * 3^4 = 2 + 0 + 9 + 0 + 0 = 11 in decimal.
    """
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


def convert_code(code: np.uint8, guess: str) -> list[Predicate]:
    """Recover the list of predicates from the 8-bit response code (see make_response_code)."""
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
    def observe(self, guess: str, predicates: list[Predicate]):
        raise NotImplementedError

    @abc.abstractmethod
    def guess(self) -> str:
        raise NotImplementedError


class RandomValidPolicy(Policy):
    def __init__(self, word_lists: dict[str, list[str]], list_='solutions'):
        self.word_lists = word_lists
        self.predicates = []
        self.list_ = list_

    def observe(self, guess: str, predicates: Clue):
        self.predicates += predicates

    def guess(self) -> str:
        valid_guesses = [w for w in self.word_lists[self.list_] if all(p.satisfies(w) for p in self.predicates)]
        return random.choice(valid_guesses)


class GreedyHeuristicPolicy(Policy):
    """Guesses the word that minimizes the expected uncertainty about the solution, looking one step ahead.
    
    Formally, let $C(G, S)$ be the clue obtained by guessing $G$ when the solution is $S$.
    We represent a clue by the list of words that are incompatible with the guess.
    Then, for a sequence of clues $C_1, C_2, \\ldots, C_n$, the remaining uncertinty is given by
    $U(C_1, C_2, \\ldots, C_n) = U_0 - |\\Cup_i C_i|$.

    The greedy heuristic policy guesses the word $G$ that maximizes the expected uncertainty
    $E[U(C_1, C_2, \\ldots, C_n, C(G_{n+1}, S))]$ over $G_{n+1}$, where the randomness is over the possible solutions $S$.
    """
    def __init__(self, word_lists: WordLists, clue_matrix: np.ndarray, hard_mode=False, n_attempts=6):
        self.word_lists = word_lists
        self.clue_matrix = clue_matrix
        self.predicates: list[Predicate] = []  # flat list of predicates
        self.hard_mode = hard_mode
        self.remaining = n_attempts

    def observe(self, guess: str, predicates: Clue):
        self.predicates += predicates
        self.remaining -= 1

    def guess(self) -> str:
        guess_list = self.word_lists['guesses']
        solution_list = self.word_lists['solutions']
        possible = np.array([all(p.satisfies(w) for p in self.predicates) for w in solution_list], dtype=np.bool_)
        if self.remaining >= possible.sum():  # limit guesses to valid solutions
            solutions = [s for (p, s) in zip(possible, solution_list) if p]
            guess_idcs = [i for i, g in enumerate(guess_list) if g in solutions]
            guess_list = [guess_list[i] for i in guess_idcs]
            possible_clue_matrix = self.clue_matrix[guess_idcs][:, possible]
        elif self.hard_mode:  # limit guesses to valid
            guess_idcs = [i for i, g in enumerate(guess_list) if all(p.satisfies(g) for p in self.predicates)]
            guess_list = [guess_list[i] for i in guess_idcs]
            possible_clue_matrix = self.clue_matrix[guess_idcs][:, possible]
        else:
            possible_clue_matrix = self.clue_matrix[:, possible]
        # pcm_decimal = np.sum([possible_clue_matrix[:, :, i] * 3**i for i in range(5)], axis=0)
        clue_counts = [np.unique(row, return_counts=True)[1] for row in possible_clue_matrix]
        entropies = [scipy.stats.entropy(row) for row in clue_counts]
        action_idx = np.argmax(entropies)
        return guess_list[action_idx]


class InteractivePolicy(Policy):
    def __init__(self, word_lists: WordLists, dark_mode=True):
        self.word_lists = word_lists
        self.dark_mode = dark_mode

    def guess(self) -> str:
        s = input('> ')
        while s not in self.word_lists['guesses']:
            print("Not in word list. Try again")
            s = input('> ')
        return s

    def observe(self, guess: str, predicates: Clue):
        s = "".join(p.emoji(dark_mode=self.dark_mode) for p in predicates)
        print(s)


class Game:
    def __init__(self, word_lists: WordLists, solution=None, hard_mode=False, n_attempts=6, policy=None):
        self.word_lists = word_lists
        self.solution = solution or random.choice(self.word_lists['solutions'])
        self.hard_mode = hard_mode
        self.policy = policy or RandomValidPolicy(word_lists)
        self.n_attempts = n_attempts

    def play(self) -> tuple[list[str], list[Clue]]:
        guesses = []
        clues = []
        for _t in range(self.n_attempts):
            guess = self.policy.guess()
            violates_hard_mode = self.hard_mode and not all(all(p.satisfies(guess) for p in clue) for clue in clues)
            if guess not in self.word_lists['guesses'] or violates_hard_mode:
                raise ValueError(f"Invalid guess: {guess}")
            clue = make_predicates(guess, self.solution)
            self.policy.observe(guess, clue)
            guesses.append(guess)
            clues.append(clue)
            if guess == self.solution:
                break
        return guesses, clues


def new_policy(policy_name: str, word_lists: WordLists, hard_mode: bool, clue_matrix: Optional[np.ndarray] = None, dark_mode=True) -> Policy:
    if policy_name == "interactive":
        policy = InteractivePolicy(word_lists, dark_mode=dark_mode)
    elif policy_name == "randomValid":
        policy = RandomValidPolicy(word_lists)
    elif policy_name == "greedyHeuristic":
        policy = GreedyHeuristicPolicy(word_lists, clue_matrix=clue_matrix, hard_mode=hard_mode)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    return policy


def main():
    parser = argparse.ArgumentParser(description="Solve a Wordle.")
    parser.add_argument('--policy', help="the policy to use", default='interactive')
    parser.add_argument('--batch', help="run multiple wordles from the word list", default=None)
    parser.add_argument('--hard-mode', help="run in hard mode", action='store_true')
    parser.add_argument('--word-list', help="the word list to use", default='words.json')
    parser.add_argument('--dark-mode', help="dark mode", action='store_true')
    parser.add_argument('solution', help="the correct solution", default=None)

    args = parser.parse_args()

    print("Loading word lists...")
    with open(args.word_list, 'r', encoding='utf-8') as fp:
        word_lists = json.load(fp)
        word_lists['guesses'] += word_lists['solutions']
        # TODO: why this filtering?
        word_lists['solutions'] = [w for w in word_lists['solutions'] if all(w[i] not in w[i + 1:] for i in range(4))]

    try:
        print("Loading clue matrix")
        clue_matrix = np.load('clue_matrix.npy')
    except IOError:
        print("Computing clue matrix")
        clue_matrix = np.zeros((len(word_lists['guesses']), len(word_lists['solutions'])), dtype=np.uint8)
        iterator = it.product(enumerate(word_lists['guesses']), enumerate(word_lists['solutions']))
        for (i, a), (j, w) in tqdm(iterator, total=len(word_lists['guesses']) * len(word_lists['solutions'])):
            clue_matrix[i, j] = make_response_code(a, w)
        np.save('clue_matrix.npy', clue_matrix)
        print("Made clue matrix")

    if not args.batch:
        policy = new_policy(args.policy, word_lists, hard_mode=args.hard_mode, clue_matrix=clue_matrix, dark_mode=args.dark_mode)

        game = Game(word_lists, policy=policy, solution=args.solution, hard_mode=args.hard_mode)
        guesses, clues = game.play()
        clue_strs = ["".join(p.emoji() for p in preds) for preds in clues]
        print(f"Wordle {game.solution} {len(guesses)}/{game.n_attempts}")
        for guess, clue_str in zip(guesses, clue_strs):
            print(f"{guess}\t{clue_str}")
    else:
        if args.batch == 'all':
            solutions = word_lists['solutions']
        else:
            solutions = random.sample(word_lists['solutions'], int(args.batch))
        results = {'statistics': defaultdict(int), 'games': []}
        for _r, solution in enumerate(tqdm(solutions)):
            policy = new_policy(args.policy, word_lists, hard_mode=args.hard_mode, clue_matrix=clue_matrix)
            game = Game(word_lists, policy=policy, solution=solution, hard_mode=args.hard_mode)
            guesses, clues = game.play()
            results['games'].append({'solution': solution, 'guesses': guesses})
            if guesses[-1] == game.solution:
                results['statistics'][len(guesses)] += 1
            else:
                results['statistics'][-1] += 1
        print("Saving results")
        with open('results.json', mode='w', encoding='utf-8') as fp:
            json.dump(results, fp, indent=2, ensure_ascii=False)
        print("Statistics:")
        stats = {k: round(v / sum(results['statistics'].values()), 3) for k, v in results['statistics'].items()}
        print(stats)


if __name__ == '__main__':
    main()
