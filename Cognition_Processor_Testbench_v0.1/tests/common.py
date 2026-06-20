from pathlib import Path

from cptb.ccdl import CCDLParser
from cptb.seam import SemanticContract

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "programs" / "cognition_processor_v0_1.ccdl"
CONTRACT = ROOT / "programs" / "semantic_contracts.json"


def load_baseline():
    return CCDLParser.parse_file(PROGRAM)


def load_contract():
    return SemanticContract.load(CONTRACT)
