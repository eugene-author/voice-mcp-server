"""Project conventions that linters cannot express: one-line docstrings, no comments, Mock* naming, host probing."""

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = ROOT / "app" / "migrations" / "versions"
PYTHON_FILES = sorted(
    path
    for path in [*(ROOT / "app").rglob("*.py"), *(ROOT / "tests").rglob("*.py")]
    if not path.is_relative_to(EXCLUDED)
)
TEST_TEXT_FILES = sorted(
    path
    for path in (ROOT / "tests").rglob("*")
    if path.is_file() and path.suffix in {".py", ".md", ".txt", ".json", ".yaml", ".yml"}
)
ALLOWED_PRAGMA = re.compile(r"^# (type: ignore\[[a-z0-9-]+(, *[a-z0-9-]+)*\]|noqa: [A-Z]+[0-9]+(, *[A-Z]+[0-9]+)*)$")
DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
FORBIDDEN_WORD = re.compile("fa" + "ke", re.IGNORECASE)
PROSE_TEXT_FILES = sorted(
    path
    for path in [
        *(ROOT / "app").rglob("*"),
        *(ROOT / "tests").rglob("*"),
        *(ROOT / "docs").rglob("*"),
        ROOT / "README.md",
    ]
    if path.is_file() and path.suffix in {".py", ".md", ".txt", ".json", ".yaml", ".yml"}
)
RETIRED_WORD = re.compile("halluc" + "inat", re.IGNORECASE)
ABSOLUTE_WORD = re.compile(r"\bne" + r"ver\b", re.IGNORECASE)
ALL_PYTHON_FILES = sorted([*(ROOT / "app").rglob("*.py"), *(ROOT / "tests").rglob("*.py")])
HF_MODULE = "huggingface_hub"
HF_INSTALLER = "app/adapters/models/huggingface/catalog.py"
HF_NAMES = frozenset({"HfApi", "snapshot_download"})
SNAPSHOT_KEYWORDS = frozenset({"repo_id", "revision", "local_dir"})
MODEL_INFO_KEYWORDS = frozenset({"revision"})


def docstring_problems(source: str) -> list[str]:
    """Return every definition whose docstring is missing or spans more than one line."""
    problems: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, DOCUMENTED):
            continue
        docstring = ast.get_docstring(node, clean=False)
        name = getattr(node, "name", "<module>")
        if docstring is None:
            problems.append(f"{name}: missing docstring")
        elif "\n" in docstring.strip():
            problems.append(f"{name}: docstring spans several lines")
    return problems


def comment_problems(source: str) -> list[str]:
    """Return every # comment that is not a narrow type: ignore[code] or noqa: CODE pragma."""
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [
        f"line {token.start[0]}: {token.string}"
        for token in tokens
        if token.type == tokenize.COMMENT and not ALLOWED_PRAGMA.match(token.string)
    ]


def _ids(paths: list[Path]) -> list[str]:
    """Return readable test IDs."""
    return [str(path.relative_to(ROOT)) for path in paths]


@pytest.mark.parametrize("path", PYTHON_FILES, ids=_ids(PYTHON_FILES))
def test_every_definition_has_a_one_line_docstring(path: Path) -> None:
    """Require a one-line docstring on every module, class, function, method and property."""
    problems = docstring_problems(path.read_text(encoding="utf-8"))
    assert not problems, problems


@pytest.mark.parametrize("path", PYTHON_FILES, ids=_ids(PYTHON_FILES))
def test_no_comments_other_than_narrow_pragmas(path: Path) -> None:
    """Reject # comments except type: ignore[code] and noqa: CODE pragmas."""
    problems = comment_problems(path.read_text(encoding="utf-8"))
    assert not problems, problems


def test_tests_never_use_the_forbidden_test_double_word() -> None:
    """Name test doubles Mock* and keep the forbidden word out of everything under tests/."""
    offenders = [str(path.relative_to(ROOT)) for path in TEST_TEXT_FILES if FORBIDDEN_WORD.search(path.name)]
    offenders.extend(
        f"{path.relative_to(ROOT)}:{number}"
        for path in TEST_TEXT_FILES
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FORBIDDEN_WORD.search(line)
    )
    assert not offenders, offenders


def retired_word_problems(text: str) -> list[int]:
    """Return the line numbers that use the retired word for transcript quality flags."""
    return [number for number, line in enumerate(text.splitlines(), start=1) if RETIRED_WORD.search(line)]


@pytest.mark.parametrize("path", PROSE_TEXT_FILES, ids=_ids(PROSE_TEXT_FILES))
def test_text_files_do_not_use_the_retired_word(path: Path) -> None:
    """Reject the retired word in every text file under app, tests and docs, and in the README."""
    assert not retired_word_problems(path.read_text(encoding="utf-8"))


def test_retired_word_check_rejects_a_broken_sample() -> None:
    """Flag the retired word whatever its case, and accept the words that replaced it."""
    broken = "clean line\n" + "A HALLUC" + "INATION check\ntext_in_silence and segment_flags\n"
    assert retired_word_problems(broken) == [2]


def docstring_lines(source: str) -> list[tuple[int, str]]:
    """Return the line and text of each module, class, function and attribute docstring."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body):
            if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)):
                continue
            if not isinstance(statement.value.value, str):
                continue
            first = index == 0 and isinstance(node, DOCUMENTED)
            attribute = index > 0 and isinstance(body[index - 1], ast.Assign | ast.AnnAssign)
            if first or attribute:
                found.append((statement.lineno, statement.value.value))
    return found


def absolute_word_problems(source: str) -> list[int]:
    """Return the lines of docstrings that use the absolute word as a whole word, in any case."""
    return [line for line, text in docstring_lines(source) if ABSOLUTE_WORD.search(text)]


@pytest.mark.parametrize("path", ALL_PYTHON_FILES, ids=_ids(ALL_PYTHON_FILES))
def test_docstrings_do_not_use_the_absolute_word(path: Path) -> None:
    """Require docstrings to state what happens, including migration docstrings."""
    assert not absolute_word_problems(path.read_text(encoding="utf-8"))


def test_absolute_word_check_rejects_a_broken_sample() -> None:
    """Flag module, class, function and attribute docstrings, and ignore strings and longer words."""
    word = "Ne" + "ver"
    broken = (
        f'"""{word} one."""\n\n\nclass Sample:\n    """{word.lower()} two."""\n\n    size = 1\n'
        f'    """{word.upper()} three."""\n\n    def run(self):\n        """{word} four."""\n'
        f'        return "{word} in a value"\n\n\ndef other():\n    """Nevertheless fine."""\n'
    )
    assert sorted(absolute_word_problems(broken)) == [1, 5, 8, 11]


def test_checkers_reject_a_broken_sample() -> None:
    """Flag a missing docstring, a multi-line docstring and a plain comment, and accept narrow pragmas."""
    broken = '"""Module."""\n\n\ndef missing():\n    return 1\n\n\ndef long():\n    """First.\n\n    More.\n    """\n'
    commented = "value = 1  # explain\nother = 2  # type: ignore[assignment]\nthird = 3  # noqa: S101\n"
    assert docstring_problems(broken) == ["missing: missing docstring", "long: docstring spans several lines"]
    assert docstring_problems('"""Module.\n\nDetails.\n"""\n') == ["<module>: docstring spans several lines"]
    assert comment_problems(commented) == ["line 1: # explain"]
    assert FORBIDDEN_WORD.search("A " + "FA" + "KE store") is not None


def huggingface_hub_problems(relative_path: str, source: str) -> list[str]:
    """Return huggingface_hub imports and calls outside the subset shared by its 0.x and 1.x+ releases."""
    problems: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import | ast.ImportFrom):
            problems.extend(_hf_import_problems(relative_path, node))
        elif isinstance(node, ast.Call):
            problems.extend(_hf_call_problems(node))
    return problems


def _hf_import_problems(relative_path: str, node: ast.Import | ast.ImportFrom) -> list[str]:
    """Return problems with an import statement that touches huggingface_hub."""
    if isinstance(node, ast.ImportFrom):
        modules, names = [node.module or ""], {alias.name for alias in node.names}
    else:
        modules, names = [alias.name for alias in node.names], set()
    if not any(module.split(".")[0] == HF_MODULE for module in modules):
        return []
    if relative_path != HF_INSTALLER:
        return [f"line {node.lineno}: {HF_MODULE} imported outside {HF_INSTALLER}"]
    if modules != [HF_MODULE] or not names or names - HF_NAMES:
        return [f"line {node.lineno}: only 'from {HF_MODULE} import {', '.join(sorted(HF_NAMES))}' is allowed"]
    return []


def _hf_call_problems(call: ast.Call) -> list[str]:
    """Return problems with a snapshot_download or model_info call's arguments."""
    func = call.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
    keywords = {keyword.arg or "**" for keyword in call.keywords}
    if name == "snapshot_download" and (call.args or keywords - SNAPSHOT_KEYWORDS):
        return [f"line {call.lineno}: snapshot_download takes only {sorted(SNAPSHOT_KEYWORDS)} keywords"]
    if name == "model_info" and (len(call.args) != 1 or keywords - MODEL_INFO_KEYWORDS):
        return [f"line {call.lineno}: model_info takes only the repo ID and revision"]
    return []


def test_huggingface_hub_usage_works_on_both_major_versions() -> None:
    """Limit huggingface_hub use to calls that behave the same on 0.36 (tts lock) and 2.x (mlx lock)."""
    offenders = [
        f"{path.relative_to(ROOT)}: {problem}"
        for path in sorted((ROOT / "app").rglob("*.py"))
        for problem in huggingface_hub_problems(str(path.relative_to(ROOT)), path.read_text(encoding="utf-8"))
    ]
    assert not offenders, offenders


def test_huggingface_hub_checker_rejects_a_broken_sample() -> None:
    """Flag a stray import, an extra imported name and arguments outside the shared subset."""
    installer = (
        "from huggingface_hub import HfApi, snapshot_download, hf_hub_download\n"
        "snapshot_download(repo_id='r', revision='v', local_dir='d', local_dir_use_symlinks=False)\n"
        "HfApi().model_info('r', revision='v', files_metadata=True)\n"
        "import huggingface_hub\n"
    )
    assert len(huggingface_hub_problems(HF_INSTALLER, installer)) == 4
    assert huggingface_hub_problems("app/cli.py", "from huggingface_hub import HfApi\n") == [
        f"line 1: {HF_MODULE} imported outside {HF_INSTALLER}"
    ]
    accepted = (
        "from huggingface_hub import HfApi, snapshot_download\n"
        "snapshot_download(repo_id='r', revision='v', local_dir='d')\n"
        "HfApi().model_info('r', revision='v')\n"
    )
    assert huggingface_hub_problems(HF_INSTALLER, accepted) == []


def _probe_calls(tree: ast.AST) -> list[str]:
    """Return platform and CUDA probing calls in a syntax tree."""
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
        if (owner, node.func.attr) == ("platform", "system") or node.func.attr == "get_cuda_device_count":
            found.append(f"{owner}.{node.func.attr}")
    return found


def test_platform_checks_only_in_host_adapter() -> None:
    """Allow platform.system() and CUDA probing calls only in app/adapters/host/."""
    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "app").rglob("*.py")
        if "adapters/host" not in str(path) and _probe_calls(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not offenders, offenders
