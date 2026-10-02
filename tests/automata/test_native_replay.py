"""Persistent native replay catalog and whole-game sampling contracts."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from automata.decision import DecisionDescriptor
from automata.harness.game_runner import DEFAULT_MAP
from automata.models.contracts import CURRENT_MAP_SCHEMA_VERSION
from automata.models.shared_encoder.artifacts import (
    ArtifactFile,
    ArtifactTensor,
    Gen1ModelArtifactManifest,
    ModelArtifactManifest,
)
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.observation import encode_decision
from automata.observation.hero_adapters import HeroObservationAdapterRegistry
from automata.runtime.effects import register_all_effects
from automata.training.native_dataset import (
    NativeGameIdentity,
    PolicyDatasetRecord,
    native_game_id,
    native_sample_id,
    publish_native_game,
)
from automata.training.native_indexed_dataset import (
    IndexedNativeDataset,
    build_native_indexed_dataset,
    create_native_source_receipt,
    create_native_source_receipt_from_completions,
)
from automata.training.native_receipts import (
    NativeDatasetCompletionReceipt,
    NativeGameCompletionReceipt,
)
from automata.training.native_replay import (
    NativeReplayCatalog,
    NativeReplayCompatibility,
    NativeReplayConfig,
    NativeReplayGameRef,
    NativeReplayGeneration,
    load_native_replay_catalog,
    sample_native_replay,
    update_native_replay_catalog,
)
from automata.training.native_splits import (
    NativeSeedRange,
    NativeSplitConfig,
    create_native_split_ledger,
    extend_native_split_ledger,
)
from automata.training.search_targets import SearchActionTarget, SearchPolicyTarget
from goa2.data.heroes import HeroRegistry
from goa2.domain.models import GameType, TeamColor
from goa2.engine.setup import GameSetup


def _split_config() -> NativeSplitConfig:
    return NativeSplitConfig(
        namespace="native-replay-tests",
        salt="fixed-test-salt",
        validation_fraction=0.2,
        seed_ranges=(NativeSeedRange(purpose="training", start=0, stop=10_000),),
    )


def _config(*, max_games: int | None = None) -> NativeReplayConfig:
    return NativeReplayConfig(split_config=_split_config(), max_games=max_games)


def _compatibility() -> NativeReplayCompatibility:
    decision = TensorFeatureSchema.current()
    value = StableValueTensorSchema.current()
    return NativeReplayCompatibility(
        decision_tensor_schema_id=decision.schema_id,
        decision_tensor_schema_version=decision.schema_version,
        decision_tensor_schema_digest=decision.digest,
        stable_value_tensor_schema_id=value.schema_id,
        stable_value_tensor_schema_version=value.schema_version,
        stable_value_tensor_schema_digest=value.digest,
    )


def _identity(
    seed: int,
    *,
    generation: str = "generation-1",
    source_model_digest: str | None = None,
) -> NativeGameIdentity:
    values: dict[str, Any] = {
        "world_seed": seed,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": generation,
        "source_revision": "abc123",
        "dirty_tree_hash": "clean",
        "source_model_digest": source_model_digest,
        "search_config_id": "search-1",
        "generator_config_id": "generator-1",
    }
    return NativeGameIdentity(game_id=native_game_id(**values), **values)


def _policy(game: NativeGameIdentity) -> PolicyDatasetRecord:
    register_all_effects()
    state = GameSetup.create_game(
        DEFAULT_MAP,
        list(game.red_composition),
        list(game.blue_composition),
        game_type=game.game_type,
        seed=game.world_seed,
    )
    hero = state.teams[TeamColor.RED].heroes[0]
    hero.hand = hero.hand[:1]
    observation = encode_decision(
        state,
        DecisionDescriptor("CARD", hero=hero),
        tuple(card.id for card in hero.hand),
        decision_owner_hero_id=str(hero.id),
        perspective_team=TeamColor.RED.value,
    )
    candidate = observation.candidates[0]
    return PolicyDatasetRecord(
        game=game,
        sample_id=native_sample_id(
            game_id=game.game_id,
            sample_kind="POLICY",
            sample_index=0,
        ),
        sample_index=0,
        policy_index=0,
        perspective_team="RED",
        observation=observation,
        target=SearchPolicyTarget(
            actions=(
                SearchActionTarget(
                    candidate=candidate,
                    prior_probability=1.0,
                    sample_count=1,
                    mean_value=0.0,
                    value_variance=0.0,
                    improved_probability=1.0,
                    selected=True,
                ),
            )
        ),
    )


def _indexed_generation(
    root: Path,
    *,
    identities: tuple[NativeGameIdentity, ...],
) -> tuple[IndexedNativeDataset, NativeDatasetCompletionReceipt]:
    source = root / "source"
    games: list[NativeGameCompletionReceipt] = []
    for ordinal, identity in enumerate(identities):
        logical_name = f"game-{ordinal}.jsonl"
        path = source / logical_name
        publish_native_game(path, (_policy(identity),))
        payload = path.read_bytes()
        games.append(
            NativeGameCompletionReceipt(
                logical_name=logical_name,
                game=identity,
                file_sha256=hashlib.sha256(payload).hexdigest(),
                file_size=len(payload),
                row_count=1,
                policy_row_count=1,
                value_row_count=0,
                boundary_count=0,
                reason="game_over",
                terminal_winner="RED",
            )
        )
    completion = NativeDatasetCompletionReceipt(games=tuple(games))
    inventory = create_native_source_receipt_from_completions(source, completion)
    inventory_path = root / "inventory.json"
    inventory_path.write_bytes(inventory.canonical_bytes())
    dataset = build_native_indexed_dataset(source, inventory_path, root / "cache", chunk_size=1)
    return dataset, completion


def _parent(digest: str) -> Gen1ModelArtifactManifest:
    decision = TensorFeatureSchema.current()
    value = StableValueTensorSchema.current()
    adapters = HeroObservationAdapterRegistry()
    heroes = tuple(HeroRegistry.list_heroes())
    versions = {
        "generic": adapters.generic_version,
        **{
            hero: adapters.registered_versions.get(hero, adapters.generic_version)
            for hero in heroes
        },
    }
    maps_root = Path(__file__).parents[2] / "src" / "goa2" / "data" / "maps"
    files = {
        name: ArtifactFile(length=1, sha256="f" * 64)
        for name in ("decision_schema.json", "stable_value_schema.json", "weights.pt")
    }
    return Gen1ModelArtifactManifest(
        model_digest=digest,
        map_schema_version=CURRENT_MAP_SCHEMA_VERSION,
        hero_adapter_versions=versions,
        supported_heroes=heroes,
        supported_maps=tuple(sorted(path.stem for path in maps_root.glob("*.json"))),
        supported_game_types=tuple(game_type.value for game_type in GameType),
        decision_tensor_schema_id=decision.schema_id,
        decision_tensor_schema_version=decision.schema_version,
        decision_tensor_schema_digest=decision.digest,
        stable_value_tensor_schema_id=value.schema_id,
        stable_value_tensor_schema_version=value.schema_version,
        stable_value_tensor_schema_digest=value.digest,
        architecture_config={"architecture": "test"},
        tensors={"weight": ArtifactTensor(shape=(1,), dtype="float32")},
        files=files,
    )


def _train_seeds(config: NativeSplitConfig, count: int) -> tuple[int, ...]:
    ledger = create_native_split_ledger(config)
    seeds: list[int] = []
    for seed in range(10_000):
        candidate = extend_native_split_ledger(ledger, (seed,))
        if candidate.split_for_seed(seed) == "train":
            ledger = candidate
            seeds.append(seed)
            if len(seeds) == count:
                return tuple(seeds)
    raise AssertionError("test split config did not produce enough train seeds")


def _catalog(*, game_count: int = 3, max_games: int | None = None) -> NativeReplayCatalog:
    config = _config(max_games=max_games)
    seeds = _train_seeds(config.split_config, game_count)
    ledger = extend_native_split_ledger(create_native_split_ledger(config.split_config), seeds)
    identities = tuple(_identity(seed) for seed in seeds)
    all_ids = tuple(game.game_id for game in identities)
    generation = NativeReplayGeneration(
        generation_id="generation-1",
        source_model_digest=None,
        dataset_digest="a" * 64,
        source_digest="b" * 64,
        completion_receipt_digest="c" * 64,
        all_game_ids=all_ids,
        replay_game_ids=all_ids,
    )
    first_retained = max(0, game_count - max_games) if max_games is not None else 0
    games = tuple(
        NativeReplayGameRef(
            game=game,
            dataset_digest=generation.dataset_digest,
            source_digest=generation.source_digest,
            completion_receipt_digest=generation.completion_receipt_digest,
            source_logical_name=f"game-{index}.jsonl",
            policy_row_count=0 if index == 0 else index,
            value_row_count=1 if index in {0, 2} else 0,
            boundary_count=1 if index in {0, 2} else 0,
            insertion_index=index,
        )
        for index, game in enumerate(identities)
        if index >= first_retained
    )
    return NativeReplayCatalog(
        config=config,
        compatibility=_compatibility(),
        split_ledger=ledger,
        generations=(generation,),
        games=games,
    )


def test_catalog_round_trips_canonically_and_samples_whole_games(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "catalog.json"
    path.write_bytes(catalog.canonical_bytes())

    loaded = load_native_replay_catalog(path)
    first = sample_native_replay(loaded, game_count=2, sampling_seed=77)
    second = sample_native_replay(loaded, game_count=2, sampling_seed=77)

    assert loaded == catalog
    assert first == second
    assert len(first.game_ids) == len(set(first.game_ids)) == 2
    assert set(first.game_ids) <= set(game.game.game_id for game in catalog.games)
    assert first.policy_contributing_game_count == sum(
        game.policy_row_count > 0 for game in first.games
    )
    assert first.value_contributing_game_count == sum(
        game.value_row_count > 0 for game in first.games
    )
    assert all(isinstance(game, NativeReplayGameRef) for game in first.games)


def test_catalog_rejects_history_ledger_capacity_and_reference_tampering() -> None:
    catalog = _catalog(game_count=3, max_games=2)
    assert len(catalog.games) == 2
    assert len(catalog.generations[0].all_game_ids) == 3
    assert len(catalog.split_ledger.assignments) == 3

    payload = catalog.model_dump(mode="python")
    payload["games"] = tuple(reversed(payload["games"]))
    with pytest.raises(ValidationError, match=r"retained|insertion|history"):
        NativeReplayCatalog.model_validate(payload, strict=True)

    payload = catalog.model_dump(mode="python")
    payload["generations"][0]["dataset_digest"] = "d" * 64
    with pytest.raises(ValidationError, match=r"digest|generation"):
        NativeReplayCatalog.model_validate(payload, strict=True)

    payload = catalog.model_dump(mode="python")
    payload["games"][0]["game"]["world_seed"] = 9_999
    with pytest.raises(ValidationError, match=r"game_id|ledger|split"):
        NativeReplayCatalog.model_validate(payload, strict=True)


def test_catalog_and_sampling_revalidate_unsafe_model_copies(tmp_path: Path) -> None:
    catalog = _catalog()
    broken = catalog.model_copy(update={"games": tuple(reversed(catalog.games))})
    with pytest.raises(ValueError, match=r"retained|insertion|history"):
        sample_native_replay(broken, game_count=1, sampling_seed=1)

    first_assignment = catalog.split_ledger.assignments[0]
    reassigned = first_assignment.model_copy(
        update={"split": "validation" if first_assignment.split == "train" else "train"}
    )
    broken_ledger = catalog.split_ledger.model_copy(
        update={"assignments": (reassigned, *catalog.split_ledger.assignments[1:])}
    )
    broken = catalog.model_copy(update={"split_ledger": broken_ledger})
    with pytest.raises(ValueError, match=r"assignment|split recipe"):
        sample_native_replay(broken, game_count=1, sampling_seed=1)

    path = tmp_path / "catalog.json"
    path.write_bytes(catalog.canonical_bytes() + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        load_native_replay_catalog(path)


def test_update_persists_bootstrap_history_ledger_and_whole_game_capacity(
    tmp_path: Path,
) -> None:
    config = _config(max_games=1)
    seed = _train_seeds(config.split_config, 1)[0]
    catalog_path = tmp_path / "catalog.json"

    first_dataset, first_completion = _indexed_generation(
        tmp_path / "first",
        identities=(_identity(seed, generation="generation-1"),),
    )
    first = update_native_replay_catalog(
        catalog_path,
        config=config,
        dataset=first_dataset,
        completion_receipt=first_completion,
        parent_artifact=None,
    )
    second_dataset, second_completion = _indexed_generation(
        tmp_path / "second",
        identities=(_identity(seed, generation="generation-2"),),
    )
    second = update_native_replay_catalog(
        catalog_path,
        config=config,
        dataset=second_dataset,
        completion_receipt=second_completion,
        parent_artifact=None,
    )

    assert len(first.games) == 1
    assert len(second.games) == 1
    assert second.games[0].game.generation_id == "generation-2"
    assert len(second.generations) == 2
    assert len(second.split_ledger.assignments) == 1
    assert second.split_ledger.split_for_seed(seed) == "train"
    assert load_native_replay_catalog(catalog_path) == second


def test_reenrolling_evicted_generation_preserves_exact_catalog_bytes(
    tmp_path: Path,
) -> None:
    config = _config(max_games=1)
    first_seed, second_seed = _train_seeds(config.split_config, 2)
    first_dataset, first_completion = _indexed_generation(
        tmp_path / "first",
        identities=(_identity(first_seed, generation="generation-1"),),
    )
    second_dataset, second_completion = _indexed_generation(
        tmp_path / "second",
        identities=(_identity(second_seed, generation="generation-2"),),
    )
    path = tmp_path / "catalog.json"
    first = update_native_replay_catalog(
        path,
        config=config,
        dataset=first_dataset,
        completion_receipt=first_completion,
        parent_artifact=None,
    )
    second = update_native_replay_catalog(
        path,
        config=config,
        dataset=second_dataset,
        completion_receipt=second_completion,
        parent_artifact=None,
    )
    assert first.games[0].game.game_id not in {game.game.game_id for game in second.games}
    before = path.read_bytes()

    with pytest.raises(ValueError, match=r"generation already exists|game IDs already exist"):
        update_native_replay_catalog(
            path,
            config=config,
            dataset=first_dataset,
            completion_receipt=first_completion,
            parent_artifact=None,
        )

    assert path.read_bytes() == before


def test_update_rejects_mixed_generation_and_source_model_identity(tmp_path: Path) -> None:
    config = _config()
    first_seed, second_seed = _train_seeds(config.split_config, 2)
    mixed_generations, mixed_generation_completion = _indexed_generation(
        tmp_path / "mixed-generations",
        identities=(
            _identity(first_seed, generation="generation-1"),
            _identity(second_seed, generation="generation-2"),
        ),
    )
    with pytest.raises(ValueError, match="exactly one generation ID"):
        update_native_replay_catalog(
            tmp_path / "mixed-generations.json",
            config=config,
            dataset=mixed_generations,
            completion_receipt=mixed_generation_completion,
            parent_artifact=None,
        )

    mixed_models, mixed_model_completion = _indexed_generation(
        tmp_path / "mixed-models",
        identities=(
            _identity(first_seed, generation="generation-3"),
            _identity(
                second_seed,
                generation="generation-3",
                source_model_digest="8" * 64,
            ),
        ),
    )
    with pytest.raises(ValueError, match="exactly one source model digest"):
        update_native_replay_catalog(
            tmp_path / "mixed-models.json",
            config=config,
            dataset=mixed_models,
            completion_receipt=mixed_model_completion,
            parent_artifact=None,
        )

    assert not (tmp_path / "mixed-generations.json").exists()
    assert not (tmp_path / "mixed-models.json").exists()


def test_bootstrap_and_exact_gen1_learned_generations_share_catalog(tmp_path: Path) -> None:
    config = _config()
    bootstrap_seed, learned_seed = _train_seeds(config.split_config, 2)
    bootstrap, bootstrap_completion = _indexed_generation(
        tmp_path / "bootstrap",
        identities=(_identity(bootstrap_seed, generation="bootstrap"),),
    )
    parent_digest = "7" * 64
    learned, learned_completion = _indexed_generation(
        tmp_path / "learned",
        identities=(
            _identity(
                learned_seed,
                generation="learned",
                source_model_digest=parent_digest,
            ),
        ),
    )
    path = tmp_path / "catalog.json"
    update_native_replay_catalog(
        path,
        config=config,
        dataset=bootstrap,
        completion_receipt=bootstrap_completion,
        parent_artifact=None,
    )

    catalog = update_native_replay_catalog(
        path,
        config=config,
        dataset=learned,
        completion_receipt=learned_completion,
        parent_artifact=_parent(parent_digest),
    )

    assert tuple(generation.source_model_digest for generation in catalog.generations) == (
        None,
        parent_digest,
    )


def test_existing_catalog_tensor_incompatibility_preserves_exact_bytes(
    tmp_path: Path,
) -> None:
    existing = _catalog()
    assert existing.compatibility is not None
    compatibility = existing.compatibility.model_copy(
        update={"decision_tensor_schema_digest": "0" * 64}
    )
    incompatible = existing.model_copy(update={"compatibility": compatibility})
    path = tmp_path / "catalog.json"
    path.write_bytes(incompatible.canonical_bytes())
    before = path.read_bytes()
    seed = _train_seeds(existing.config.split_config, 1)[0]
    dataset, completion = _indexed_generation(
        tmp_path / "new-generation",
        identities=(_identity(seed, generation="generation-2"),),
    )

    with pytest.raises(ValueError, match="incompatible"):
        update_native_replay_catalog(
            path,
            config=existing.config,
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=None,
        )

    assert path.read_bytes() == before


def test_update_enrolls_validation_seed_in_ledger_but_never_replay(
    tmp_path: Path,
) -> None:
    split = NativeSplitConfig(
        namespace="validation-ledger-test",
        salt="fixed",
        validation_fraction=0.2,
        seed_ranges=(
            NativeSeedRange(purpose="training", start=0, stop=100),
            NativeSeedRange(purpose="validation", start=100, stop=101),
        ),
    )
    config = NativeReplayConfig(split_config=split)
    train_seed = _train_seeds(split, 1)[0]
    identities = (
        _identity(train_seed, generation="mixed-generation"),
        _identity(100, generation="mixed-generation"),
    )
    dataset, completion = _indexed_generation(tmp_path / "mixed", identities=identities)

    catalog = update_native_replay_catalog(
        tmp_path / "catalog.json",
        config=config,
        dataset=dataset,
        completion_receipt=completion,
        parent_artifact=None,
    )

    assert catalog.split_ledger.split_for_seed(train_seed) == "train"
    assert catalog.split_ledger.split_for_seed(100) == "validation"
    assert catalog.generations[0].all_game_ids == tuple(game.game_id for game in identities)
    assert catalog.generations[0].replay_game_ids == (identities[0].game_id,)
    assert tuple(game.game.game_id for game in catalog.games) == (identities[0].game_id,)


def test_update_accepts_only_exact_gen1_parent_and_is_atomic_on_rejection(
    tmp_path: Path,
) -> None:
    config = _config()
    seed = _train_seeds(config.split_config, 1)[0]
    parent_digest = "9" * 64
    dataset, completion = _indexed_generation(
        tmp_path / "learned",
        identities=(
            _identity(
                seed,
                generation="learned-generation",
                source_model_digest=parent_digest,
            ),
        ),
    )
    catalog_path = tmp_path / "catalog.json"

    catalog = update_native_replay_catalog(
        catalog_path,
        config=config,
        dataset=dataset,
        completion_receipt=completion,
        parent_artifact=_parent(parent_digest),
    )
    before = catalog_path.read_bytes()
    assert catalog.generations[0].source_model_digest == parent_digest

    wrong_parent = _parent("8" * 64)
    with pytest.raises(ValueError, match="parent"):
        update_native_replay_catalog(
            catalog_path,
            config=config,
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=wrong_parent,
        )
    assert catalog_path.read_bytes() == before

    legacy = ModelArtifactManifest.model_construct()
    with pytest.raises((TypeError, ValueError), match=r"Gen1|parent"):
        update_native_replay_catalog(
            tmp_path / "legacy.json",
            config=config,
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=legacy,  # type: ignore[arg-type]
        )
    assert not (tmp_path / "legacy.json").exists()


@pytest.mark.parametrize(
    "parent_update",
    [
        {"decision_tensor_schema_digest": "7" * 64},
        {"supported_maps": ("forgotten_island",)},
        {"hero_adapter_versions": {"generic": 1}},
        {"map_schema_version": CURRENT_MAP_SCHEMA_VERSION + 1},
    ],
)
def test_update_rejects_gen1_schema_and_scope_copies_before_publication(
    tmp_path: Path,
    parent_update: dict[str, object],
) -> None:
    config = _config()
    seed = _train_seeds(config.split_config, 1)[0]
    parent_digest = "6" * 64
    dataset, completion = _indexed_generation(
        tmp_path / "generation",
        identities=(_identity(seed, generation="learned", source_model_digest=parent_digest),),
    )
    parent = _parent(parent_digest).model_copy(update=parent_update)
    path = tmp_path / "catalog.json"

    with pytest.raises((TypeError, ValueError, ValidationError), match=r"schema|scope|adapter|map"):
        update_native_replay_catalog(
            path,
            config=config,
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=parent,
        )
    assert not path.exists()


@pytest.mark.parametrize("purpose", ["evaluation", "arena", "screen", "promotion"])
def test_update_rejects_forbidden_seed_and_raw_inventory_without_catalog(
    tmp_path: Path,
    purpose: str,
) -> None:
    split = NativeSplitConfig(
        namespace="forbidden-test",
        salt="fixed",
        validation_fraction=0.2,
        seed_ranges=(NativeSeedRange(purpose=purpose, start=500, stop=501),),  # type: ignore[arg-type]
    )
    config = NativeReplayConfig(split_config=split)
    dataset, completion = _indexed_generation(
        tmp_path / "arena",
        identities=(_identity(500, generation="arena-generation"),),
    )
    path = tmp_path / "catalog.json"

    with pytest.raises(ValueError, match=rf"{purpose}|eligible|replay"):
        update_native_replay_catalog(
            path,
            config=config,
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=None,
        )
    assert not path.exists()

    raw = create_native_source_receipt(dataset.source_root, ("game-0.jsonl",))
    with pytest.raises((TypeError, ValueError, ValidationError), match=r"completion|receipt"):
        update_native_replay_catalog(
            path,
            config=config,
            dataset=dataset,
            completion_receipt=raw,  # type: ignore[arg-type]
            parent_artifact=None,
        )
    assert not path.exists()


def test_concurrent_first_updates_never_observe_a_partial_lock_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    first_seed, second_seed = _train_seeds(config.split_config, 2)
    first_dataset, first_completion = _indexed_generation(
        tmp_path / "first",
        identities=(_identity(first_seed, generation="generation-1"),),
    )
    second_dataset, second_completion = _indexed_generation(
        tmp_path / "second",
        identities=(_identity(second_seed, generation="generation-2"),),
    )
    catalog_path = tmp_path / "catalog.json"
    lock_path = tmp_path / ".catalog.json.lock"
    first_lock_entry_created = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    guard = threading.Lock()
    blocked = False
    real_open = os.open

    def coordinated_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        nonlocal blocked
        descriptor = real_open(path, flags, mode)
        candidate = Path(path)
        is_lock_initialization = candidate == lock_path or (
            candidate.parent == lock_path.parent
            and candidate.name.startswith(f".{lock_path.name}.")
            and candidate.name.endswith(".tmp")
        )
        should_block = False
        if is_lock_initialization and flags & os.O_EXCL:
            with guard:
                if not blocked:
                    blocked = True
                    should_block = True
        if should_block:
            first_lock_entry_created.set()
            if not release_first.wait(timeout=10):
                raise AssertionError("timed out coordinating first lock publication")
        return descriptor

    monkeypatch.setattr(os, "open", coordinated_open)
    errors: list[BaseException] = []

    def update_first() -> None:
        try:
            update_native_replay_catalog(
                catalog_path,
                config=config,
                dataset=first_dataset,
                completion_receipt=first_completion,
                parent_artifact=None,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def update_second() -> None:
        try:
            update_native_replay_catalog(
                catalog_path,
                config=config,
                dataset=second_dataset,
                completion_receipt=second_completion,
                parent_artifact=None,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            second_done.set()

    first_thread = threading.Thread(target=update_first)
    second_thread = threading.Thread(target=update_second)
    first_thread.start()
    assert first_lock_entry_created.wait(timeout=10)
    second_thread.start()
    assert second_done.wait(timeout=20)
    release_first.set()
    first_thread.join(timeout=20)
    second_thread.join(timeout=20)

    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert errors == []
    assert {
        generation.generation_id
        for generation in load_native_replay_catalog(catalog_path).generations
    } == {
        "generation-1",
        "generation-2",
    }


def test_update_rejects_config_change_without_touching_catalog(tmp_path: Path) -> None:
    config = _config()
    seed = _train_seeds(config.split_config, 1)[0]
    dataset, completion = _indexed_generation(
        tmp_path / "generation",
        identities=(_identity(seed),),
    )
    path = tmp_path / "catalog.json"
    update_native_replay_catalog(
        path,
        config=config,
        dataset=dataset,
        completion_receipt=completion,
        parent_artifact=None,
    )
    before = path.read_bytes()

    with pytest.raises(ValueError, match="config"):
        update_native_replay_catalog(
            path,
            config=NativeReplayConfig(split_config=config.split_config, max_games=1),
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=None,
        )
    assert path.read_bytes() == before


def test_update_never_clobbers_unrelated_catalog_lock_or_cache_paths(tmp_path: Path) -> None:
    config = _config()
    seed = _train_seeds(config.split_config, 1)[0]
    dataset, completion = _indexed_generation(
        tmp_path / "generation",
        identities=(_identity(seed),),
    )
    unrelated = tmp_path / "catalog.json"
    unrelated.write_bytes(b"important unrelated bytes")
    with pytest.raises(ValueError, match=r"catalog|invalid"):
        update_native_replay_catalog(
            unrelated,
            config=config,
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=None,
        )
    assert unrelated.read_bytes() == b"important unrelated bytes"

    locked = tmp_path / "locked.json"
    lock = tmp_path / ".locked.json.lock"
    lock.write_bytes(b"unrelated lock bytes")
    with pytest.raises(ValueError, match=r"lock|unrelated"):
        update_native_replay_catalog(
            locked,
            config=config,
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=None,
        )
    assert lock.read_bytes() == b"unrelated lock bytes"
    assert not locked.exists()

    with pytest.raises(ValueError, match="cache"):
        update_native_replay_catalog(
            dataset.cache_dir / "catalog.json",
            config=config,
            dataset=dataset,
            completion_receipt=completion,
            parent_artifact=None,
        )


@pytest.mark.parametrize("value", [True, 0, -1])
def test_config_requires_strict_positive_capacity(value: object) -> None:
    with pytest.raises((TypeError, ValueError, ValidationError)):
        NativeReplayConfig(split_config=_split_config(), max_games=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("game_count", [True, 0, -1, 4])
def test_sampling_rejects_invalid_or_unavailable_game_counts(game_count: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        sample_native_replay(
            _catalog(), game_count=game_count, sampling_seed=1  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("sampling_seed", [True, 1.5, "1"])
def test_sampling_seed_is_a_strict_integer(sampling_seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        sample_native_replay(
            _catalog(), game_count=1, sampling_seed=sampling_seed  # type: ignore[arg-type]
        )
