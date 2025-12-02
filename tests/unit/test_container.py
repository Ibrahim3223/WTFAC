"""
Unit tests for DI Container.
"""

import pytest
from autoshorts.core import Container, ServiceLifetime


class DummyService:
    """Dummy service for testing."""
    call_count = 0

    def __init__(self):
        DummyService.call_count += 1
        self.instance_id = DummyService.call_count


class TestContainer:
    """Test dependency injection container."""

    def test_register_and_resolve(self, container):
        """Test basic registration and resolution."""
        container.register(DummyService, DummyService)

        service = container.resolve(DummyService)
        assert isinstance(service, DummyService)

    def test_singleton_lifetime(self, container):
        """Test singleton returns same instance."""
        DummyService.call_count = 0

        container.register(
            DummyService,
            DummyService,
            ServiceLifetime.SINGLETON
        )

        service1 = container.resolve(DummyService)
        service2 = container.resolve(DummyService)

        assert service1 is service2
        assert service1.instance_id == service2.instance_id
        assert DummyService.call_count == 1

    def test_transient_lifetime(self, container):
        """Test transient returns new instance each time."""
        DummyService.call_count = 0

        container.register(
            DummyService,
            DummyService,
            ServiceLifetime.TRANSIENT
        )

        service1 = container.resolve(DummyService)
        service2 = container.resolve(DummyService)

        assert service1 is not service2
        assert service1.instance_id != service2.instance_id
        assert DummyService.call_count == 2

    def test_register_instance(self, container):
        """Test registering an existing instance."""
        instance = DummyService()
        container.register_instance(DummyService, instance)

        resolved = container.resolve(DummyService)
        assert resolved is instance

    def test_resolve_unregistered_raises(self, container):
        """Test resolving unregistered service raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            container.resolve(DummyService)

    def test_resolve_optional(self, container):
        """Test resolve_optional returns None for unregistered."""
        result = container.resolve_optional(DummyService)
        assert result is None

    def test_is_registered(self, container):
        """Test is_registered check."""
        assert not container.is_registered(DummyService)

        container.register(DummyService, DummyService)
        assert container.is_registered(DummyService)

    def test_clear(self, container):
        """Test clearing container."""
        container.register(DummyService, DummyService)
        assert container.is_registered(DummyService)

        container.clear()
        assert not container.is_registered(DummyService)

    def test_create_scope(self, container):
        """Test creating a scoped container."""
        container.register(DummyService, DummyService)

        scope = container.create_scope()
        assert scope.is_registered(DummyService)

        # Changes to scope don't affect parent
        scope.clear()
        assert not scope.is_registered(DummyService)
        assert container.is_registered(DummyService)
