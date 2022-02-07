from __future__ import print_function

import pytest

from gym_puyopuyo import util
from gym_puyopuyo.field import TallField

_ = None
R = 0
G = 1
Y = 2
B = 3
P = 4
C = 5
W = 6


def test_gravity():
    stack = [
        R, R, _, _, _, _, _, _,
        G, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, R, _,
        _, _, _, _, _, _, R, _,
        _, _, _, _, _, _, _, G,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
    ]
    field = TallField.from_list(stack)
    field.render()
    field.handle_gravity()
    print()
    field.render()
    stack = field.to_list()
    assert (stack == [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        R, _, _, _, _, _, R, _,
        G, R, _, _, _, _, R, G,
    ])


@pytest.mark.parametrize("tsu_rules", [True, False])
def test_clear_groups(tsu_rules):
    O = Y + 1  # noqa
    stack = [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        O, _, _, _, _, _, _, _,
        R, _, _, _, _, _, _, _,
        R, _, _, _, _, _, _, _,
        R, _, _, _, _, _, _, _,
        R, _, _, _, _, _, _, _,
        R, _, _, _, _, _, _, _,
        R, _, _, _, _, _, _, _,
        R, _, _, _, _, _, _, _,
        R, _, _, _, _, _, _, _,
        R, _, G, _, _, _, _, _,
        R, _, G, _, _, _, _, _,
        R, _, G, O, O, _, Y, _,
        R, _, G, O, O, _, Y, Y,
    ]
    field = TallField.from_list(stack, tsu_rules=tsu_rules, has_garbage=True)
    field.render()
    score = field.clear_groups(5)
    print(score)
    field.render()
    stack = field.to_list()
    expected = [_] * 8 * 3
    if tsu_rules:
        expected += [O, _, _, _, _, _, _, _]
    else:
        expected += [_] * 8
    assert (score == 10 * (4 + 12) * (3 + 10 + 96))
    assert (stack == expected + [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, O, _, Y, _,
        _, _, _, _, O, _, Y, Y,
    ])


@pytest.mark.parametrize("tsu_rules", [True, False])
def test_resolve_plain(tsu_rules):
    stack = [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, G, _, _, _, _, _, _,
        _, G, _, _, _, _, _, _,
        _, G, _, _, _, _, _, _,
        _, B, G, _, _, _, _, _,
        _, G, B, _, _, _, _, _,
        _, G, B, _, _, _, _, _,
        R, R, G, _, _, _, _, _,
        R, R, G, G, _, _, _, _,
    ]
    field = TallField.from_list(stack, tsu_rules=tsu_rules)
    field.render()
    print()
    score, chain = field.resolve()
    field.render()
    stack = field.to_list()
    assert (stack == [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, B, _, _, _, _, _,
        _, B, B, _, _, _, _, _,
    ])
    assert (chain == 2)
    assert (score == 940)


def test_complex_resolve():
    stack = [
        R, G, _, _, _, _, _, _,
        _, _, R, _, _, _, _, _,
        _, _, R, _, _, _, _, _,
        _, _, R, _, _, _, _, _,
        _, _, Y, R, _, _, _, _,
        _, _, Y, R, _, _, _, _,
        _, _, Y, R, _, _, _, _,
        _, _, R, Y, _, _, _, _,
        _, _, R, Y, _, _, _, _,
        _, _, R, Y, _, _, _, _,
        _, _, Y, R, _, _, _, _,
        _, _, Y, R, _, _, _, _,
        _, _, Y, R, _, _, _, _,
        R, G, B, Y, _, _, _, _,
        R, R, G, B, _, _, _, _,
        G, G, B, B, _, _, _, _,
    ]
    field = TallField.from_list(stack)
    field.render()
    print()
    score, chain = field.resolve()
    field.render()
    stack = field.to_list()
    assert (stack == [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
    ])
    assert (chain == 3)
    expected = 4 * 10 * (1)  # Reds
    expected += 5 * 10 * (8 + 2)  # Greens
    # Blues, yellows and reds
    num_cleared = 26
    chain_power = 16
    group_bonuses = 0 + 0 + 3 + 3 + 3
    color_bonus = 6
    expected += num_cleared * 10 * (chain_power + group_bonuses + color_bonus)
    assert (score == expected)


def test_resolve_ghost():
    stack = [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, P, _, _,
        _, _, _, _, _, P, _, _,
        _, _, _, _, _, P, _, _,
        _, _, _, _, _, P, _, _,
        _, _, _, _, _, B, _, _,
        _, _, _, _, _, B, _, _,
        _, _, _, _, _, B, _, _,
        _, _, _, _, _, B, _, _,
        _, _, _, _, _, R, _, _,
        _, _, _, _, _, R, _, _,
        _, _, _, _, _, Y, _, _,
        _, _, _, _, _, G, _, _,
        _, _, _, _, _, G, _, _,
    ]
    field = TallField.from_list(stack, tsu_rules=True)
    field.render()
    print()
    score, chain = field.resolve()
    field.render()
    stack = field.to_list()
    assert (stack == [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, R, _, _,
        _, _, _, _, _, R, _, _,
        _, _, _, _, _, Y, _, _,
        _, _, _, _, _, G, _, _,
        _, _, _, _, _, G, _, _,
    ])
    assert (chain == 2)
    assert (score == 360)


def test_resolve_garbage():
    O = G + 1  # noqa
    stack = [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, O, _, _,
        _, _, _, _, O, R, _, _,
        _, _, _, _, O, R, _, _,
        _, _, _, _, O, R, _, _,
        _, _, _, _, O, R, _, _,
        _, _, _, _, O, R, _, _,
        _, _, _, _, O, R, _, _,
        _, _, _, _, O, R, _, _,
        _, _, _, _, O, R, _, _,
        _, _, _, O, O, R, _, _,
        _, _, _, O, O, R, _, _,
        _, _, _, O, O, G, O, _,
        _, _, _, O, O, G, O, O,
    ]
    field = TallField.from_list(stack, tsu_rules=True, has_garbage=True)
    field.render()
    print()
    score, chain = field.resolve()
    field.render()
    stack = field.to_list()
    assert (stack == [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, O, _, _, _, _,
        _, _, _, O, _, O, _, _,
        _, _, _, O, O, G, O, _,
        _, _, _, O, O, G, O, O,
    ])
    assert (chain == 1)
    assert (score == 700)


def test_overlay():
    field = TallField(3)
    field.overlay([_, G, R, _, _, _, _, _])
    field.handle_gravity()
    field.render()
    print()
    field.overlay([
        _, Y, _, _, _, _, _, _,
        _, Y, _, _, _, _, _, _,
    ])
    field.handle_gravity()
    field.render()
    stack = field.to_list()
    assert (stack == [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, Y, _, _, _, _, _, _,
        _, Y, _, _, _, _, _, _,
        _, G, R, _, _, _, _, _,
    ])


def test_overlapping_overlay():
    field = TallField(4)
    field.overlay([
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, R, R, _, _, _, _, _,
        R, _, _, R, G, _, _, _,
        R, _, _, _, _, _, _, _,
        R, _, _, _, _, _, _, _,
        _, R, _, _, _, _, _, _,
        _, _, R, _, _, _, _, _,
        _, _, _, R, B, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
    ])
    field.render()
    print()
    field.overlay([
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, R, R, _,
        _, _, _, Y, R, _, _, R,
        _, _, _, _, _, _, _, R,
        _, _, _, _, _, _, _, R,
        _, _, _, _, _, _, R, _,
        _, _, _, _, _, R, _, _,
        _, _, _, G, R, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
    ])
    field.render()
    stack = field.to_list()
    assert (stack == [
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, R, R, _, _, R, R, _,
        R, _, _, R, G, _, _, R,
        R, _, _, _, _, _, _, R,
        R, _, _, _, _, _, _, R,
        _, R, _, _, _, _, R, _,
        _, _, R, _, _, R, _, _,
        _, _, _, R, B, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
    ])


def test_encode():
    stack = [
        _, R, _, _, _, _, _, _,
        G, _, G, _, _, _, _, _,
        _, _, _, G, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, R,
        _, _, _, _, _, _, _, _,
        _, _, G, _, _, _, _, _,
        G, _, _, _, _, _, _, _,
        _, _, _, R, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, G,
        _, _, _, _, _, _, _, R,
    ]
    field = TallField.from_list(stack)
    field.render()
    encoded = field.encode()
    expected = [
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ]
    assert (encoded == expected).all()


def test_mirror():
    stack = [
        R, R, _, _, _, _, _, _,
        G, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, R, _,
        _, _, _, _, _, _, R, _,
        _, _, _, _, _, _, _, G,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, G,
    ]
    field = TallField.from_list(stack)
    field.render()
    field.mirror()
    print()
    field.render()
    stack = field.to_list()
    assert (stack == [
        _, _, _, _, _, _, R, R,
        _, _, _, _, _, _, _, G,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, R, _, _, _, _, _, _,
        _, R, _, _, _, _, _, _,
        G, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        G, _, _, _, _, _, _, _,
    ])


def _reference_valid_moves(lines, width):
    valid = []
    for x in range(7):
        if x >= width - 1:
            valid.append(False)
        elif (~lines[3] & (1 << x)) | (~lines[3] & (2 << x)):
            valid.append(True)
        else:
            valid.append(False)
    for x in range(8):
        if x >= width:
            valid.append(False)
        elif (lines[3] & (1 << x)):
            valid.append(False)
        else:
            valid.append(True)
    return sum(v * (1 << i) for i, v in enumerate(valid))


@pytest.mark.parametrize("width", [6, 8])
def test_valid_moves_tsu(width):
    field = TallField(1, tsu_rules=True)
    for top in range(1 << 16):
        field.data[3] = top & 255
        field.data[4] = top >> 8
        assert (_reference_valid_moves(field.data, width) == field._valid_moves(width))


def test_render_in_place():
    field = TallField(1)
    for i in range(16):
        field.data[i] = ((i + 4234) ** 3) % 256

    for i in range(20):
        print(i)
    util.print_up(16)
    print("Let's shift this a bit!", end="")
    field.render(in_place=True)
    print("hello")
    field.render(width=6, height=13)
