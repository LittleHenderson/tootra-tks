import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inversion.engine import total_inversion, DialConfig, TargetProfile, InversionMode


def test_opposite_elements():
    elements = ["B5", "D3", "C3", "D5"]
    ops = ["+T", "->", "-T"]
    res = total_inversion(elements, ops=ops, mode=InversionMode.Opposite)
    assert res["elements"][0].startswith("C")  # World B->C via WORLD_OPP
    assert res["elements"][0].endswith("6")    # Female(N5)->Male(N6)
    assert res["elements"][1].endswith("2")    # Negative(N3)->Positive(N2)
    assert res["ops"][0] == "-T"               # +T flipped


def test_reverse_causal():
    elements = ["A2", "B3", "C4"]
    ops = ["->", "->"]
    res = total_inversion(elements, ops=ops, mode=InversionMode.ReverseCausal)
    assert res["chain"] == ["C4", "B3", "A2"]
    assert res["ops"] == ["<-", "<-"]


def test_parallel_analogue_world_map():
    elements = ["B5"]
    dial = DialConfig(
        mode=InversionMode.ParallelAnalogue,
        target_profile=TargetProfile(enable=True, from_world="B", to_world="D"),
    )
    res = total_inversion(elements, ops=[], mode=dial.mode, dial=dial)
    assert res["elements"][0].startswith("D")


if __name__ == "__main__":
    test_opposite_elements()
    test_reverse_causal()
    test_parallel_analogue_world_map()
    print("All inversion engine tests passed!")
