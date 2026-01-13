from src.tks.capability import CapabilityRegistry, ToolCapability
from src.tks.config import OperationalMode


def test_check_allowed_accepts_operational_mode():
    registry = CapabilityRegistry.get_instance()
    tool_name = "test_mode_tool"
    capability = ToolCapability(
        tool_name=tool_name,
        capability_threshold=0.2,
        mode_constraints=[OperationalMode.NORMAL.value],
    )
    registry.register(capability, overwrite=True)
    try:
        assert registry.check_allowed(
            tool_name=tool_name,
            current_skill=0.9,
            mode=OperationalMode.NORMAL,
        )
    finally:
        registry.unregister(tool_name)
