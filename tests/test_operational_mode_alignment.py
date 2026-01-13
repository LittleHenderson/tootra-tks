from src.tks.config import OperationalMode as ConfigMode
from src.tks.router import OperationalMode as RouterMode


def test_operational_mode_alignment():
    assert ConfigMode is RouterMode
    assert ConfigMode.NORMAL is RouterMode.NORMAL
    assert ConfigMode.HIGH_STAKES.value == "High-Stakes"
    assert ConfigMode.CRITICAL.value == "Critical"
