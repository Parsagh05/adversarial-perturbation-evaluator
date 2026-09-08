"""What shot counts each few-shot adapter can actually run.

Not every model covers the usual 1/2/4 sweep, and the two exceptions are
properties of what their authors released rather than gaps in the adapters:
WinCLIP asserts ``k_shot in [0, 1, 5, 10]`` and ships a committed selection file
with entries only for those, and InCTRL publishes checkpoints for 2, 4 and 8
shots only. This pins the whole picture so a sweep cannot silently ask an
adapter for a count it has no reference set or checkpoint for.
"""

import inspect
import json
import re
from pathlib import Path

import pytest

from fpeval.adapters import base
import fpeval.adapters  # noqa: F401


# adapter -> the shot counts it can run, and why that set and no other.
SUPPORTED = {
    # datasets/{dataset,mvtec,visa}.py all assert k_shot in [0, 1, 5, 10], and
    # datasets/seeds_mvtec/<class>/selected_samples_per_run.txt only lists
    # "<run>-1", "<run>-5" and "<run>-10" selections.
    "winclip_fewshot": frozenset({1, 5, 10}),
    # The released Drive archive holds checkpoints/{2,4,8}/checkpoint.pyth and
    # the sample-prompt archive holds <dataset>/{2,4,8}/<class>.pt.
    "inctrl": frozenset({2, 4, 8}),
    # test.sh sweeps "for few_shot in 1 2 4".
    "afclip_fewshot": frozenset({1, 2, 4}),
    # A plain memory bank over k images; test_few_shot.sh ships k=4.
    "aprilgan_fewshot": frozenset({1, 2, 4}),
    # scripts/benchmark_few_shot.sh loops "for k in 1 2 4".
    "subspacead": frozenset({1, 2, 4}),
    # One released model per (dataset, shot) for 1, 2 and 4.
    "inpformer": frozenset({1, 2, 4}),
    # --shots is a free count of reference images.
    "fade": frozenset({1, 2, 4}),
    # --k_shot is a free count; self.shot = len(few_shot_samples).
    "univad": frozenset({1, 2, 4}),
    # fix_few_path ships fix_{1,2,4}-shot.txt (and 8/16, which are not wired up).
    "dictas": frozenset({1, 2, 4}),
    # Each test script documents a --round per shot: 1, 2 and 4.
    "kagprompt": frozenset({1, 2, 4}),
    # The released package covers 1, 2 and 4 shots per class and task.
    "promptad": frozenset({1, 2, 4}),
}
# The usual comparison grid.
SWEEP = frozenset({1, 2, 4})


def shot_parameter(name: str) -> str:
    parameters = inspect.signature(base._ADAPTERS[name].__init__).parameters
    for candidate in ("k_shot", "shot"):
        if candidate in parameters:
            return candidate
    raise AssertionError(f"{name} exposes no shot parameter")


def test_regime_matches_the_declared_few_shot_set():
    """Results are filed under the regime, so the split has to be exact.

    ``regime`` reads the constructor rather than a second list, and this pins
    that the reading agrees with the declared set over every registered name,
    aliases included.
    """
    from tests.test_notebooks import FEW_SHOT, ZERO_SHOT

    for name in FEW_SHOT:
        assert base.regime(name) == "few_shot", name
    for name in ZERO_SHOT:
        assert base.regime(name) == "zero_shot", name
    # Aliases must land in the same tree as the name they alias, or the same
    # model would write into both regimes depending on how it was spelled.
    squash = lambda value: value.replace("-", "").replace("_", "")
    few_shot = {squash(name) for name in FEW_SHOT}
    for name in base.adapter_names():
        expected = "few_shot" if squash(name) in few_shot else "zero_shot"
        assert base.regime(name) == expected, name


def test_regime_rejects_an_unknown_adapter():
    with pytest.raises(ValueError, match="Unknown adapter"):
        base.regime("not_a_model")


def test_every_few_shot_adapter_is_listed():
    """A new few-shot adapter must declare which counts it supports."""
    from tests.test_notebooks import FEW_SHOT

    assert set(SUPPORTED) == set(FEW_SHOT)


def test_declared_shot_values_match_the_supported_set():
    """Where an adapter pins SHOT_VALUES, it must agree with this table."""
    import sys

    checked = 0
    for name, supported in SUPPORTED.items():
        module = sys.modules[base._ADAPTERS[name].__module__]
        declared = getattr(module, "SHOT_VALUES", None)
        if declared is None:
            continue
        assert frozenset(declared) == supported, name
        checked += 1
    # WinCLIP, InCTRL, SubspaceAD, INP-Former, DictAS, KAG-Prompt and PromptAD
    # all pin one; the rest accept any positive count.
    assert checked >= 7


def test_only_two_models_fall_outside_the_usual_sweep():
    """The 1/2/4 grid is available everywhere except these two."""
    full = {name for name, shots in SUPPORTED.items() if SWEEP <= shots}
    partial = {name: sorted(shots) for name, shots in SUPPORTED.items()
               if not SWEEP <= shots}
    assert partial == {"winclip_fewshot": [1, 5, 10], "inctrl": [2, 4, 8]}
    assert len(full) == 9
    # Neither exception is empty at every point of the sweep, so both still
    # appear in a 1/2/4 comparison - just not at every k.
    assert SUPPORTED["winclip_fewshot"] & SWEEP == {1}
    assert SUPPORTED["inctrl"] & SWEEP == {2, 4}


def test_no_single_shot_count_covers_every_adapter():
    """Worth knowing before reading a table as if it were matched."""
    for k in SWEEP:
        missing = {name for name, shots in SUPPORTED.items() if k not in shots}
        assert missing, f"k={k} unexpectedly covers everything"
    assert {name for name, shots in SUPPORTED.items() if 1 not in shots} == {"inctrl"}
    assert {name for name, shots in SUPPORTED.items() if 2 not in shots} == {
        "winclip_fewshot"
    }
    assert {name for name, shots in SUPPORTED.items() if 4 not in shots} == {
        "winclip_fewshot"
    }


@pytest.mark.parametrize("name", sorted(SUPPORTED))
def test_shipped_config_uses_a_supported_shot_count(name):
    config = json.loads(
        Path(f"configs/{name}.example.json").read_text(encoding="utf-8")
    )
    parameter = shot_parameter(name)
    for target, kwargs in config["model_kwargs_by_target"].items():
        assert parameter in kwargs, (name, target)
        assert kwargs[parameter] in SUPPORTED[name], (name, target, kwargs[parameter])


@pytest.mark.parametrize("name", sorted(SUPPORTED))
def test_notebook_uses_a_supported_shot_count(name):
    notebook = json.loads(
        Path(f"notebooks/few_shot/kaggle_{name}.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    found = re.findall(r"^(?:K_SHOT|SHOT)\s*=\s*(\d+)", source, re.M)
    assert len(found) == 1, (name, found)
    assert int(found[0]) in SUPPORTED[name], (name, found[0])


@pytest.mark.parametrize("name", sorted(SUPPORTED))
def test_the_shot_count_reaches_the_results(name):
    """Every adapter records its shot count, so a mixed sweep stays readable."""
    source = Path(
        inspect.getfile(base._ADAPTERS[name])
    ).read_text(encoding="utf-8")
    parameter = shot_parameter(name)
    assert f'"{parameter}": self.{parameter}' in source, name
