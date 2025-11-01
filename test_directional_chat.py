"""
Simple test script for directional chat engines.

Tests basic functionality without requiring trained models.
"""

from nanochat.directional_chat_engine import (
    DirectionalChatEngine,
    ForwardChatEngine,
    BackwardChatEngine,
    BidirectionalChatEngine,
    create_chat_engine
)


def test_base_class():
    """Test base class structure."""
    # Should not be able to instantiate directly
    try:
        engine = DirectionalChatEngine(None, None, 'forward')
        engine.add_user_message("test")
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass
    print("✅ Base class enforces implementation")


def test_factory():
    """Test factory function logic (without instantiation)."""
    # We can't fully test without a real tokenizer, but we can verify imports
    # and class types exist
    assert ForwardChatEngine is not None
    assert BackwardChatEngine is not None
    assert BidirectionalChatEngine is not None
    assert create_chat_engine is not None

    # Test direction validation logic by checking the classes
    assert hasattr(ForwardChatEngine, 'can_toggle_direction')
    assert hasattr(BackwardChatEngine, 'can_toggle_direction')
    assert hasattr(BidirectionalChatEngine, 'can_toggle_direction')

    print("✅ Factory function and classes exist correctly")


def test_direction_toggle():
    """Test direction toggle method existence."""
    # Check that toggle methods exist on the base class
    assert hasattr(DirectionalChatEngine, 'can_toggle_direction')
    assert hasattr(DirectionalChatEngine, 'set_direction')

    # Check all implementations have these methods
    assert hasattr(ForwardChatEngine, 'can_toggle_direction')
    assert hasattr(BackwardChatEngine, 'can_toggle_direction')
    assert hasattr(BidirectionalChatEngine, 'can_toggle_direction')

    print("✅ Direction toggle methods exist correctly")


if __name__ == '__main__':
    print("Testing directional chat engines...")
    test_base_class()
    test_factory()
    test_direction_toggle()
    print("\n✅ All tests passed!")
