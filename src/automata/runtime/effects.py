"""Register all hero card effects.

Mirrors `goa2.server.app.register_all_effects` but without importing the FastAPI
server, so headless AI/self-play doesn't pull in web dependencies. Card effects
register themselves as an import side effect (the `@register_effect` decorator).
"""

from __future__ import annotations

import importlib
import pkgutil

_registered = False


def register_all_effects() -> None:
    """Import every goa2.scripts.*_effects module and every hero module once."""
    global _registered
    if _registered:
        return

    # Hero definitions (populate the HeroRegistry).
    import goa2.data.heroes as heroes_pkg

    for mod in pkgutil.iter_modules(heroes_pkg.__path__):
        if mod.name not in ("registry", "__init__"):
            importlib.import_module(f"goa2.data.heroes.{mod.name}")

    # Effect scripts (populate the effect registry via @register_effect).
    import goa2.scripts as scripts_pkg

    for mod in pkgutil.iter_modules(scripts_pkg.__path__):
        if mod.name.endswith("_effects"):
            importlib.import_module(f"goa2.scripts.{mod.name}")

    _registered = True
