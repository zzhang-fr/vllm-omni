# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion attention backend registry.

Two ways to extend the registry:

1. **Override a built-in slot** — replace `FLASH_ATTN`, `TORCH_SDPA`,
   `SAGE_ATTN`, or `CUSTOM_ATTN` with your own implementation:

       @register_diffusion_backend(DiffusionAttentionBackendEnum.FLASH_ATTN)
       class MyFlashAttn: ...

2. **Register a new named backend** — for third-party plugins that
   want their own discoverable name (instead of overriding a built-in):

       register_diffusion_backend("my_kernel", MyKernelBackend)

   End users then select it via per-role config:

       --diffusion-attention-config.per_role.self.backend=my_kernel

   Built-in names always win on collision; registering a name that
   matches a built-in raises ValueError.
"""

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


class _DiffusionBackendEnumMeta(EnumMeta):
    """Metaclass for DiffusionAttentionBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str) -> "DiffusionAttentionBackendEnum":
        """Get backend by name with helpful error messages."""
        try:
            return super().__getitem__(name)  # type: ignore[return-value]
        except KeyError:
            members = list(cls.__members__.keys())
            registered = sorted(_DIFFUSION_ATTN_REGISTRY.keys())
            hint = ", ".join(members)
            if registered:
                hint += "; or registered plugins: " + ", ".join(registered)
            raise ValueError(
                f"Unknown diffusion attention backend: '{name}'. Valid options are: {hint}"
            ) from None


class DiffusionAttentionBackendEnum(Enum, metaclass=_DiffusionBackendEnumMeta):
    """Enumeration of built-in diffusion attention backends.

    Each member's value is the default class path; overrides are stored in
    `_DIFFUSION_ATTN_OVERRIDES`. ``CUSTOM_ATTN`` has no built-in class —
    overriding it is one way for a plugin to take over the conventional
    "custom" slot. New plugin names that don't shadow a built-in should
    use :func:`register_diffusion_backend` with a string name instead.

    Example:
        # Get a built-in
        backend_cls = DiffusionAttentionBackendEnum.FLASH_ATTN.get_class()

        # Override a built-in slot
        @register_diffusion_backend(DiffusionAttentionBackendEnum.FLASH_ATTN)
        class MyFlashAttn: ...

        # Register a new named plugin
        register_diffusion_backend("my_kernel", MyKernelBackend)
    """

    # Common backends (available on most platforms)
    FLASH_ATTN = "vllm_omni.diffusion.attention.backends.flash_attn.FlashAttentionBackend"
    TORCH_SDPA = "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"
    SAGE_ATTN = "vllm_omni.diffusion.attention.backends.sage_attn.SageAttentionBackend"

    # Reserved slot for a plugin that wants the canonical "custom" name.
    # Has no built-in class path — must be overridden via
    # register_diffusion_backend(DiffusionAttentionBackendEnum.CUSTOM_ATTN, ...)
    # before use.
    CUSTOM_ATTN = ""

    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If backend has empty path and is not registered
        """
        path = _DIFFUSION_ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_diffusion_backend(DiffusionAttentionBackendEnum.{self.name}, "
                f"'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If backend has empty path and is not registered
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override
        """
        return self in _DIFFUSION_ATTN_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _DIFFUSION_ATTN_OVERRIDES.pop(self, None)


# Per-enum-member override paths.
_DIFFUSION_ATTN_OVERRIDES: dict[DiffusionAttentionBackendEnum, str] = {}

# Registry of plugin-supplied backends keyed by case-insensitive name.
# Values are fully-qualified class paths.
_DIFFUSION_ATTN_REGISTRY: dict[str, str] = {}


def _class_path(cls_or_path: "type | str") -> str:
    if isinstance(cls_or_path, str):
        return cls_or_path
    return f"{cls_or_path.__module__}.{cls_or_path.__qualname__}"


def _validate_backend_cls(cls_or_path: "type | str", name: str) -> None:
    """Verify the resolved object is a usable AttentionBackend subclass.

    Run at registration time so plugin authors fail loud immediately,
    not at first forward(). When called with a class directly, the
    check skips the qualname round-trip — this matters for closures
    (e.g. test fixtures) whose qualified name doesn't import.
    """
    from vllm_omni.diffusion.attention.backends.abstract import (
        AttentionBackend,
        AttentionImpl,
    )

    if isinstance(cls_or_path, str):
        try:
            cls = resolve_obj_by_qualname(cls_or_path)
        except Exception as e:
            raise ValueError(
                f"register_diffusion_backend({name!r}): failed to import '{cls_or_path}': {e}"
            ) from e
    else:
        cls = cls_or_path

    if not (isinstance(cls, type) and issubclass(cls, AttentionBackend)):
        raise TypeError(
            f"register_diffusion_backend({name!r}): {cls!r} is "
            f"{type(cls).__name__}, not a subclass of AttentionBackend."
        )
    impl_cls = cls.get_impl_cls()
    if not (isinstance(impl_cls, type) and issubclass(impl_cls, AttentionImpl)):
        raise TypeError(
            f"register_diffusion_backend({name!r}): {cls.__name__}.get_impl_cls() returned "
            f"{type(impl_cls).__name__}, not a subclass of AttentionImpl."
        )


def register_diffusion_backend(
    name_or_backend: "str | DiffusionAttentionBackendEnum",
    cls_or_path: "type | str | None" = None,
    *,
    validate: bool = True,
) -> Callable[[type], type] | None:
    """Register a diffusion attention backend.

    Two modes:

    1. **Override a built-in slot** — pass a `DiffusionAttentionBackendEnum`
       member as the first argument. Replaces the default class path for
       that slot.

    2. **Register a new named plugin** — pass a `str` as the first
       argument. The name must not shadow any built-in enum member;
       case-insensitive duplicate registrations are rejected with a
       warning (first-wins).

    The second argument can be:
        - A class (its module path is taken).
        - A fully-qualified class path string (e.g. ``"my.module.MyBackend"``).
        - ``None`` — returns a decorator that captures the decorated class.

    Args:
        name_or_backend: Built-in enum member to override, OR a new
            string name to register.
        cls_or_path: Implementation class, its qualified path, or
            ``None`` (decorator form).
        validate: If True (default), verify the resolved class is a
            subclass of :class:`AttentionBackend` whose
            ``get_impl_cls()`` returns an :class:`AttentionImpl`
            subclass. Set False to defer validation (e.g. when the
            class isn't importable yet at registration call site).

    Returns:
        Decorator if ``cls_or_path`` is None; otherwise None.

    Examples:
        # Override a built-in
        @register_diffusion_backend(DiffusionAttentionBackendEnum.FLASH_ATTN)
        class MyFlashAttn: ...

        # Register a new plugin by name (direct)
        register_diffusion_backend("my_kernel", MyKernelBackend)

        # Register a new plugin by name (decorator)
        @register_diffusion_backend("my_kernel")
        class MyKernelBackend(AttentionBackend): ...

        # Register by string path (no import at call site)
        register_diffusion_backend("my_kernel", "my.pkg.MyKernelBackend",
                                  validate=False)
    """
    # --- Mode 1: override a built-in enum slot ---
    if isinstance(name_or_backend, DiffusionAttentionBackendEnum):
        slot = name_or_backend

        def _record(cls_or_p: "type | str") -> None:
            if validate:
                _validate_backend_cls(cls_or_p, slot.name)
            _DIFFUSION_ATTN_OVERRIDES[slot] = _class_path(cls_or_p)

        if cls_or_path is None:
            def decorator(cls: type) -> type:
                _record(cls)
                return cls

            return decorator

        _record(cls_or_path)
        return None

    # --- Mode 2: register a new named plugin ---
    if isinstance(name_or_backend, str):
        name = name_or_backend
        if not name:
            raise ValueError("register_diffusion_backend: name cannot be empty.")
        if name.upper() in DiffusionAttentionBackendEnum.__members__:
            raise ValueError(
                f"register_diffusion_backend({name!r}): '{name}' shadows built-in "
                f"DiffusionAttentionBackendEnum.{name.upper()}. To replace the built-in, "
                f"pass the enum member instead: "
                f"register_diffusion_backend(DiffusionAttentionBackendEnum.{name.upper()}, ...)"
            )
        key = name.lower()

        def _record(cls_or_p: "type | str") -> None:
            if validate:
                _validate_backend_cls(cls_or_p, name)
            path = _class_path(cls_or_p)
            if key in _DIFFUSION_ATTN_REGISTRY:
                existing = _DIFFUSION_ATTN_REGISTRY[key]
                if existing == path:
                    return  # idempotent
                logger.warning(
                    "register_diffusion_backend(%r): name already registered to %s; "
                    "keeping the first registration (ignoring %s).",
                    name,
                    existing,
                    path,
                )
                return
            _DIFFUSION_ATTN_REGISTRY[key] = path

        if cls_or_path is None:
            def decorator(cls: type) -> type:
                _record(cls)
                return cls

            return decorator

        _record(cls_or_path)
        return None

    raise TypeError(
        f"register_diffusion_backend: first argument must be a "
        f"DiffusionAttentionBackendEnum or str; got {type(name_or_backend).__name__}."
    )


def resolve_registered_backend(name: str) -> str | None:
    """Look up a plugin-registered backend by name (case-insensitive).

    Returns the fully-qualified class path, or None if not registered.
    """
    return _DIFFUSION_ATTN_REGISTRY.get(name.lower())
