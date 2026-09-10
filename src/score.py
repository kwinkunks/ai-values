import numpy as np


def _y002(answer: str) -> int:
    """
    Scoring for Y002 (aims of the country, pick two):
    options 1+3 → score 1 (traditional/order), 2+4 → score 3 (post-materialist), else 2.
    """
    a, b = sorted(map(int, answer.split(',')))
    if (a, b) == (1, 3):
        return 1
    elif (a, b) == (2, 4):
        return 3
    return 2


def _y003(answer: str) -> int:
    """
    Scoring for Y003 (qualities important for children to learn):
    +1 for each of independence/determination, -1 for each of faith/obedience, clamped to [-2, 2].
    Matches the quality names in English (EN), bokmål (NO), and Georgian (KA), since the
    answer is free text in whichever language the question was asked. Georgian is matched
    on stems (the language is heavily inflected, so answers may add case endings):
    დამოუკიდებლ(ობა) independence, გამბედაობ/შეუპოვრ determination-perseverance,
    რელიგ(იური რწმენა) religious faith, მორჩილ(ება) obedience. ('religi' rather than a
    bare 'tro'/'რწმენ' to avoid false substring hits.)
    """
    x = (
        ('independence' in answer or 'selvstendighet' in answer or 'დამოუკიდებლ' in answer)
        + ('determination' in answer or 'besluttsomhet' in answer or 'გამბედაობ' in answer or 'შეუპოვრ' in answer)
        - ('faith' in answer or 'religi' in answer or 'რელიგ' in answer)
        - ('obedience' in answer or 'lydighet' in answer or 'მორჩილ' in answer)
    )
    return max(min(int(x), 2), -2)


def compute_score(variable: str, answer: str) -> int | float:
    """Score a single question response. Returns NaN for unrecognised variables or unparseable answers."""
    scorers = {
        'F063': lambda x: max(min(int(x), 10), 1),
        'Y003': _y003,
        'F120': lambda x: max(min(int(x), 10), 1),
        'G006': lambda x: max(min(int(x), 4), 1),
        'E018': lambda x: max(min(int(x), 3), 1),
        'Y002': _y002,
        'A008': lambda x: max(min(int(x), 4), 1),
        'F118': lambda x: max(min(int(x), 10), 1),
        'E025': lambda x: {'a': 1, 'b': 2, 'c': 3}.get(x, np.nan),
        'A165': lambda x: {'a': 1, 'b': 2}.get(x, np.nan),
    }
    return scorers.get(variable, lambda x: np.nan)(answer)
