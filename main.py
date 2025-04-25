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
    """Guesses the word with the highest clue entropy.
    
    Formally, let $C(G, S)$ be the clue obtained by guessing $G$ when the solution is $S$.
    A clue is given by a list of colors (green, yellow, black), one for each of the five letters in the guess.
    The corresponding clue number is this list interpreted as a number in base-3, i.e., a number between 0 and 3^5 - 1.
    The clue entropy for $G$ is the entropy of the distribution of clue numbers $H[C(G, S)]$ with the randomness taken over $S$.
    """
    def __init__(self, word_lists: WordLists, clue_matrix: np.ndarray, hard_mode=False, n_attempts=6):
        """Params:
        word_lists: dict[str, list[str]]: word lists. The 'solutions' list must be a subset of the 'guesses' list.
        clue_matrix: np.ndarray: clue matrix. The rows are guesses and the columns are solutions.
        hard_mode: bool: if True, the policy will only guess words that are compatible with the predicates.
        n_attempts: int: number of total attempts.
        """
        guess_list_sorted = sorted(word_lists['guesses'])
        solution_list_sorted = sorted(word_lists['solutions'])
        solution_set = set(solution_list_sorted)
        self.possible_guesses = np.ones(len(guess_list_sorted), dtype=np.bool_)
        # tracks which solutions are still possible, gets updated with each guess
        self.possible_solutions = np.zeros(len(guess_list_sorted), dtype=np.bool_)
        for i, w in enumerate(guess_list_sorted):
            self.possible_solutions[i] = w in solution_set
        # mask of solution words within the guess list
        self.initial_solution_mask = self.possible_solutions.copy()
        self.guess_list_sorted = guess_list_sorted
        self.clue_matrix = clue_matrix
        self.hard_mode = hard_mode
        self.remaining = n_attempts

    def observe(self, guess: str, predicates: Clue):
        for i, w in enumerate(self.guess_list_sorted):
            if self.possible_solutions[i]:
                self.possible_solutions[i] = all(p.satisfies(w) for p in predicates)
            if self.possible_guesses[i]:
                self.possible_guesses[i] = all(p.satisfies(w) for p in predicates)
        self.remaining -= 1

    def guess(self) -> str:
        n_possible_solutions = np.sum(self.possible_solutions, dtype=int)
        if self.remaining >= n_possible_solutions:  # we are guaranteed to win if we pick a possible solution
            guess_candidates = self.possible_solutions
        elif self.hard_mode:  # we must pick a possible guess
            guess_candidates = self.possible_guesses
        else:  # anything goes
            guess_candidates = np.ones(len(self.guess_list_sorted), dtype=np.bool_)

        # convert mask over guess list to mask over solution list
        mask_within_solutions = self.possible_solutions[self.initial_solution_mask]
        clue_submatrix = self.clue_matrix[guess_candidates][:, mask_within_solutions]
        clue_counts = [np.unique(row, return_counts=True)[1] for row in clue_submatrix]
        entropies = [scipy.stats.entropy(row) for row in clue_counts]
        action_idx = np.argmax(entropies)

        # convert action_idx to index in guess_list_sorted
        guess_idx = np.where(guess_candidates)[0][action_idx]

        return self.guess_list_sorted[guess_idx]
        # n_guess_candidates = np.sum(guess_candidates, dtype=int)
        # guess_candidate_list = [w for w, p in zip(self.guess_list_sorted, guess_candidates) if p]
        # possible_solutions_list = [w for w, p in zip(self.guess_list_sorted, self.possible_solutions) if p]

        # # How much would each guess-solution pair reduce the uncertainty about the solution?
        # gs_uncertainty_reductions = np.zeros(clue_submatrix.shape, dtype=np.uint32)
        # for gi, si in tqdm(it.product(range(n_guess_candidates), range(n_possible_solutions)),
        #                    total=n_guess_candidates * n_possible_solutions):
        #     clue = clue_submatrix[gi, si]
        #     preds = convert_code(clue, guess_candidate_list[gi])
        #     for possible_solution in possible_solutions_list:
        #         if not all(p.satisfies(possible_solution) for p in preds):
        #             gs_uncertainty_reductions[gi, si] += 1

        # expected_uncertainty_reductions = np.sum(gs_uncertainty_reductions, axis=1)
        # action_idx = np.argmax(expected_uncertainty_reductions)
        # return guess_candidate_list[action_idx]


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


def run(word_list: str, solution: str, policy: str, hard_mode: bool, batch: Optional[str], dark_mode: bool):
    print("Loading word lists...")
    with open(word_list, 'r', encoding='utf-8') as fp:
        word_lists = json.load(fp)
        word_lists['guesses'] += word_lists['solutions']
        word_lists['guesses'] = sorted(set(word_lists['guesses']))
        word_lists['solutions'] = sorted(set(word_lists['solutions']))
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

    if not batch:
        policy_obj = new_policy(policy, word_lists, hard_mode=hard_mode, clue_matrix=clue_matrix, dark_mode=dark_mode)

        game = Game(word_lists, policy=policy_obj, solution=solution, hard_mode=hard_mode)
        guesses, clues = game.play()
        clue_strs = ["".join(p.emoji() for p in preds) for preds in clues]
        print(f"Wordle {game.solution} {len(guesses)}/{game.n_attempts}")
        for guess, clue_str in zip(guesses, clue_strs):
            print(f"{guess}\t{clue_str}")
    else:
        if batch == 'all':
            solutions = word_lists['solutions']
        else:
            solutions = random.sample(word_lists['solutions'], int(batch))
        results = {'statistics': defaultdict(int), 'games': []}
        for _r, solution in enumerate(tqdm(solutions)):
            policy_obj = new_policy(policy, word_lists, hard_mode=hard_mode, clue_matrix=clue_matrix)
            game = Game(word_lists, policy=policy_obj, solution=solution, hard_mode=hard_mode)
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


def main():
    parser = argparse.ArgumentParser(description="Solve a Wordle.")
    parser.add_argument('--policy', help="the policy to use", default='interactive')
    parser.add_argument('--batch', help="run multiple wordles from the word list", default=None)
    parser.add_argument('--hard-mode', help="run in hard mode", action='store_true')
    parser.add_argument('--word-list', help="the word list to use", default='words.json')
    parser.add_argument('--dark-mode', help="dark mode", action='store_true')
    parser.add_argument('solution', help="the correct solution", default=None)

    args = parser.parse_args()

    print(vars(args))
    run(**vars(args))


def debug():
    run('words_sv.json', 'svara', 'greedyHeuristic', hard_mode=False, batch=None, dark_mode=False)


if __name__ == '__main__':
    main()
    # debug()
