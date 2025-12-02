"""
Lightweight Dependency Injection Container.

Provides simple service registration and resolution for better testability
and decoupling.
"""

from typing import TypeVar, Generic, Dict, Callable, Any, Optional, Type
import threading

T = TypeVar('T')


class ServiceLifetime:
    """Service lifetime options."""
    SINGLETON = "singleton"  # One instance for the entire application
    TRANSIENT = "transient"  # New instance each time


class Container:
    """
    Simple dependency injection container.

    Example:
        # Create container
        container = Container()

        # Register services
        container.register(
            GeminiClient,
            lambda: GeminiClient(api_key="..."),
            lifetime=ServiceLifetime.SINGLETON
        )

        # Resolve service
        gemini = container.resolve(GeminiClient)

        # For testing, replace with mock
        container.register(GeminiClient, lambda: MockGeminiClient())
    """

    def __init__(self):
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lifetimes: Dict[Type, str] = {}
        self._lock = threading.Lock()

    def register(
        self,
        interface: Type[T],
        factory: Callable[[], T],
        lifetime: str = ServiceLifetime.SINGLETON
    ) -> None:
        """
        Register a service with its factory function.

        Args:
            interface: The service type/interface
            factory: Function that creates the service instance
            lifetime: ServiceLifetime.SINGLETON or ServiceLifetime.TRANSIENT
        """
        with self._lock:
            self._factories[interface] = factory
            self._lifetimes[interface] = lifetime
            # Clear cached singleton if re-registering
            if interface in self._singletons:
                del self._singletons[interface]

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """
        Register an already-created instance as a singleton.

        Args:
            interface: The service type/interface
            instance: The service instance
        """
        with self._lock:
            self._singletons[interface] = instance
            self._lifetimes[interface] = ServiceLifetime.SINGLETON

    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve a service by its interface/type.

        Args:
            interface: The service type to resolve

        Returns:
            The service instance

        Raises:
            KeyError: If service not registered
        """
        if interface not in self._factories and interface not in self._singletons:
            raise KeyError(
                f"Service {interface.__name__} not registered. "
                f"Use container.register({interface.__name__}, factory) first."
            )

        # Check if singleton already created
        if interface in self._singletons:
            return self._singletons[interface]

        # Get lifetime
        lifetime = self._lifetimes.get(interface, ServiceLifetime.SINGLETON)

        # Create instance
        factory = self._factories[interface]
        instance = factory()

        # Cache if singleton
        if lifetime == ServiceLifetime.SINGLETON:
            with self._lock:
                self._singletons[interface] = instance

        return instance

    def resolve_optional(self, interface: Type[T]) -> Optional[T]:
        """
        Resolve a service, returning None if not registered.

        Args:
            interface: The service type to resolve

        Returns:
            The service instance or None
        """
        try:
            return self.resolve(interface)
        except KeyError:
            return None

    def is_registered(self, interface: Type) -> bool:
        """Check if a service is registered."""
        return interface in self._factories or interface in self._singletons

    def clear(self) -> None:
        """Clear all registrations (useful for testing)."""
        with self._lock:
            self._factories.clear()
            self._singletons.clear()
            self._lifetimes.clear()

    def create_scope(self) -> 'Container':
        """
        Create a scoped container that inherits registrations.
        Singletons are shared, but you can override registrations.
        """
        scope = Container()
        scope._factories = self._factories.copy()
        scope._singletons = self._singletons.copy()
        scope._lifetimes = self._lifetimes.copy()
        return scope


# Global container instance (optional convenience)
_global_container: Optional[Container] = None


def get_container() -> Container:
    """Get or create the global container instance."""
    global _global_container
    if _global_container is None:
        _global_container = Container()
    return _global_container


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _global_container
    _global_container = None
