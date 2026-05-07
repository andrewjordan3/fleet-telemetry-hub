# CLAUDE.md

Guidelines for Claude Code. These bias toward correctness over speed. For trivial tasks, use judgment.

## What Success Looks Like

The goal is code you'd be proud to show a senior engineer: minimal, efficient, easily understood, self-documenting, with a logical file and folder hierarchy. Every function earns its existence. Every file has a clear single purpose. Every name communicates intent without needing a comment.

Before committing any change, ask: **"Would a senior engineer accept this in code review?"** If not, rework it. Common rejections:

- Functions with too many parameters
- Duplicated logic across files
- Silent fallbacks that hide bugs
- Weakening production code to make a test pass
- "Flexibility" or "configurability" nobody asked for
- Leaving a docstring that doesn't match the code
- Adding code when the right move was to remove or restructure

That last point deserves emphasis. **The right change is often a deletion.** When functionality belongs in an existing function, don't create a new one — extend the existing one. When a function is doing too much, don't patch around it — split it. When a pattern is wrong, don't work around it — fix it. Restructuring and removing code is not just acceptable, it's preferred over accumulating workarounds.

## Think Before Coding

Before implementing, state your assumptions. If multiple interpretations exist, present them — don't pick silently. If a simpler approach exists, say so. Push back when warranted. If something is unclear, stop and ask.

When you encounter a design tradeoff (e.g., threading parameters vs. introducing a class, renaming broadly vs. adding a compatibility shim), surface it. The wrong silent choice costs more than a brief question.

## Simplicity

Write the minimum code that solves the problem. No features beyond what was asked. No abstractions for single-use code. No error handling for impossible scenarios. If you wrote 200 lines and it could be 50, rewrite it.

## Editing Existing Code

Use `str_replace` for targeted edits to existing files. Use `create_file` only for new files. Do not rewrite an entire file to change a few functions — edit surgically.

When your changes create orphans, remove what you orphaned. When you notice pre-existing issues outside the current task's scope, note them at the end of your response under **"Observations outside current scope"** — don't fix them, with the following exceptions.

**Fix without asking** (even if outside scope): typos in comments/docstrings, imports that violate ruff/isort ordering, unused imports your changes revealed, docstrings that demonstrably contradict the code they document.

**Doc-drift fixes are always in scope, including in files you would not otherwise touch.** If you notice a docstring or comment that references a deleted symbol, a renamed function, or behavior that has moved elsewhere, fix it in the same change. Stale doc references compound across prompts — each one quietly miseducates the next reader and confuses every subsequent grep. A one-line reword now beats a reference-archaeology session three prompts later.

## Sub-Agents

Use sub-agents for parallelizable tasks: renaming a symbol across many files, updating test fixtures in bulk, sweeping docstrings, or any task where each file can be handled independently. Do not use sub-agents for tasks that require tracing data flow across files sequentially — each step depends on understanding the prior step's output.

## File and Module Organization

**Files and folders are cheap. Cramming is expensive.** When in doubt, create a new file. One responsibility per file. Target 150–200 lines; split rather than extend. Never combine unrelated things to avoid creating a file.

- **`__init__.py` files:** Imports and re-exports only. No functions, no classes, no logic.
- **Re-exports:** Only `__init__.py` may re-export symbols from other modules. No other file should re-export something it didn't define.
- **Import direction:** External callers import from the package (`from mypackage import X`). Internal callers import from siblings (`from mypackage.sibling import X`).
- **Moved functions:** When moving a function to a new location, update every caller to import from the new location. Never re-export from the old location as a compatibility shim. Bad code that silently succeeds is far worse than a loud `ImportError` that guides you to the new location.

## Functions and Classes

**Maximum 5 parameters per function.** This is enforced by ruff (PLR0913). When a function exceeds 5 parameters, evaluate in this order:

1. Can the function be split so each piece needs fewer args?
2. Would a class with shared state make this clearer?
3. Do the args bundle naturally into a config or dataclass?
4. Only if none of the above work: suppress with a `noqa` and an inline justification explaining why the alternatives don't apply.

**Check before adding a parameter.** Before adding a parameter to any function, check whether it already lives on a config object, dataclass, or other container that the caller already has. If it does, pass the container — do not thread the field individually. Example: if `BootstrapConfig` already has `n_iterations`, `confidence_level`, and `random_seed`, pass `BootstrapConfig` — do not add three parameters.

**Bundle parameters that always travel together.** If three or more parameters always appear together across call sites, they are a dataclass. Create one. Do not wait for the parameter count to trigger a linter warning — the design problem exists before the symptom appears.

**Prefer functions** that do one thing and do it well. But **use classes when state makes things clearer** — multi-phase orchestration, resource management, or anywhere threading state through arguments makes the flow harder to follow. A class that holds shared artifacts as instance attributes and exposes focused methods can be dramatically clearer than pure functions passing tuples around.

Functions over ~50 lines should be split. Orchestrators orchestrate — they call other functions but contain no business logic or computation themselves.

## Type System

- Strict type hints on every parameter, return type, and variable where non-obvious.
- Container types must be specific: `list[str]` not `list`, `dict[str, float]` not `dict`.
- No `Any` without an inline justification.
- No `object` as a type annotation — use the actual type, a `Protocol`, or a `TypeVar`.
- No `from __future__ import annotations`. Fix forward references by reordering definitions or splitting modules.
- Typed containers (frozen dataclasses, `TypedDict`) over plain dicts for structured data.

## Configuration and Constants

- Runtime-configurable values → Pydantic models (`frozen=True`, `extra='forbid'`).
- Immutable data containers → `@dataclass(frozen=True, slots=True)`.
- Column names and categorical values → frozen dataclass classes or `StrEnum`.
- Config is separate from logic. No hardcoded magic numbers in functions — derive thresholds from data or pull them from config.

## Error Handling

- Catch specific, expected exceptions at the exact point they occur, with a defined recovery action.
- Unanticipated errors fail loudly. Never catch `Exception` or `BaseException` as a silencing mechanism.
- Expected conditions (insufficient data, empty results, validation failures) are not exceptions — handle them with return types (`None`, empty DataFrame, result dataclass with a status field).
- No blanket `try/except` wrapping multi-step sequences.

## Testing

- **Never modify production code to make a test pass.** If a test reveals a problem in production code, fix the production code. If a test has an incomplete fixture or wrong expectation, fix the test.
- **Never modify a test to accommodate bad production code.** If the test is right and the code is wrong, the code must change.
- **Bad code that silently succeeds is far worse than a loud failure.** A `KeyError` from a missing column is correct behavior when the column should be there. A silent fallback that hides the absence is a bug.
- Tests live in `tests/` mirroring the `src/` structure. Use pytest fixtures and parametrize.

## Data and Computation

- Vectorized pandas/numpy — no row-level loops unless mathematically unavoidable.
- Data-derived thresholds over hardcoded values wherever possible.
- `logging.getLogger(__name__)` in every module. Levels: `DEBUG` for flow, `INFO` for milestones, `ERROR` with `exc_info=True`.

## Documentation

- Full docstrings on all public functions: Args (with types), Returns, Raises, Side Effects.
- Inline comments explain **why**, not what.
- **Docstrings must match the code.** A docstring that claims behavior the code doesn't implement is worse than no docstring. If you change what a function does, update its docstring in the same edit.

## Code Style

- Verbose explicit names: `transaction_record` not `rec`. No single-letter variables.
- `match`/`case` for multi-branch dispatch over `if`/`elif` chains.
- Named helper functions over lambdas.
- `.to_numpy()` not `.values` for pandas Series/DataFrame conversion.
- Duplicated code gets extracted into shared helpers. If the same logic appears in two places, factor it out.

## Naming and Imports

- Underscore prefix = file-private. Production code never exports or imports underscore-prefixed names outside their defining module. Tests may import underscore-prefixed names from production modules to exercise private helpers directly — keeping direct coverage of non-trivial internal functions outweighs the boundary rule for test code.
- Imports organized to ruff standards (`isort`-compatible).
- `__all__: list[str]` in every module.

## Scope Discipline

When working on a task, if you notice issues outside the current prompt's scope (other than the auto-fix items listed under "Editing Existing Code"), note them under **"Observations outside current scope"** at the end of your response. Do not fix them.
