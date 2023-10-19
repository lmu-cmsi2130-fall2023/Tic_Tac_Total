"""
Artificial Intelligence responsible for playing the game of T3!
Implements the alpha-beta-pruning mini-max search algorithm
"""
from dataclasses import *
from typing import *
from t3_state import *


def choose(state: "T3State") -> Optional["T3Action"]:
    MAX = state._odd_turn
    MIN = not MAX

    # Returns values 1, -1, or 0 if win, lose, or tie
    def evaluate(state: T3State) -> int:
        if state.is_win():
            return 1 if state._odd_turn else -1
        else:
            return 0

    # Recursively calls itself until base case and then returns and evaluation for the state
    def minimax(state: T3State, depth: int, alpha: float, beta: float, player: bool) -> tuple[float, None] | tuple[
        float, T3Action]:
        if depth == 0 or state.is_win() or state.is_tie():
            return evaluate(state), None

        best_action = None
        # Mini-max function with alpha-beta pruning
        if player == MAX:
            max_eval: float = float("-inf")
            for action, next_state in state.get_transitions():
                eval, act = minimax(next_state, depth - 1, alpha, beta, MIN)
                action.minimax_score = eval
                action.depth = depth
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = float("inf")
            for action, next_state in state.get_transitions():
                eval, act = minimax(next_state, depth - 1, alpha, beta, MAX)
                action.minimax_score = eval
                action.depth = depth
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            return min_eval, best_action

    # Depth is chosen by the amount of moves left on in the state so that base case can be reached when depth = 0
    depth = 0
    for _ in state.get_open_tiles():
        depth = depth + 1
    best_utility, best_action = minimax(state, depth, float("-inf"), float("inf"), MAX)

    return best_action

