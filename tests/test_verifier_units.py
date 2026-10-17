import pytest

from gasp.verification import correct


@pytest.mark.parametrize("answer,gold", [
    ("3 bolts", "3"), ("3 square meters", "3"), ("-1.5 meters", "-1.5"),
    ("1,200 dollars", "1200"), (r"3x", "3x"), (r"2 ab + 3", "2ab+3"),
    (r"\frac{1}{2}", "0.5"), ("The robe takes a total of 3 bolts.", "3"),
])
def test_units_and_symbolic_final_answers(answer, gold):
    assert correct(f"<answer>{answer}</answer>", gold, "math_verify") == 1


@pytest.mark.parametrize("answer", ["4 bolts", "3x", "3 x", "3*x", "2ab+3"])
def test_does_not_accept_wrong_units_value_or_strip_symbolic_terms(answer):
    assert correct(f"<answer>{answer}</answer>", "3", "math_verify") == 0


def test_units_do_not_make_unfinished_reasoning_a_final_answer():
    assert correct("I considered 3 bolts but have not finished", "3", "math_verify") == 0
