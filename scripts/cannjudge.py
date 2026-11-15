#!/usr/bin/env python3
"""CANNJudge 命令行提交工具（西南赛区决赛 SinkhornNorm / npu_kernel_dev）。

用法：
  python3 scripts/cannjudge.py login
  python3 scripts/cannjudge.py submit --watch
  python3 scripts/cannjudge.py result --watch

默认从 scripts/acounts.env 读取账号（也兼容 accounts.env）。
密码只用于登录请求，不会写入 .cache。
"""

from __future__ import annotations

import argparse
import getpass
import http.cookiejar
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


BASE_URL = "https://cannjudge.cn"
DEFAULT_CONTEST = "op_challenge_xinan_final"
DEFAULT_PROBLEM = "sinkhorn_norm"

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SRC = REPO_ROOT / "submission"
CACHE_DIR = SCRIPT_DIR / ".cache"
DEFAULT_ENV_FILES = (
    SCRIPT_DIR / "acounts.env",
    SCRIPT_DIR / "accounts.env",
)

ACCOUNT_KEYS = {"email", "account", "username", "user", "login", "cannjudge_account"}
PASSWORD_KEYS = {"code", "password", "passwd", "pass", "cannjudge_password"}

PROTECTED_FILES = {"judge.asc", "data_utils.h", "main.asc", "CMakeLists.txt",
                   "reference.py", "sample.py", "requirements.txt", "run.sh", "README.md"}
KERNEL_FILE_RE = re.compile(r"^(?=.{1,128}$)[\w][\w.-]*\.(asc|h)$")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
CMAKE_UNITS_RE = re.compile(r"set\s*\(\s*SINKHORN_KERNEL_UNITS\s*(.*?)\s*\)", re.S)
CMAKE_UNIT_NAME_RE = re.compile(r"[\w.-]+\.(?:asc|h)")
SYSTEM_INCLUDES = {"kernel_operator.h"}
PENDING_STATUSES = {"Running", "Pending", "Queued", "Judging", "Waiting"}
# Internal headroom budget, not a claim about the judge's undocumented limit.
# The previously accepted request was ~367 KB; keep new requests below 320 KiB.
UPLOAD_REQUEST_BUDGET = 320 * 1024

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str, body: Any = None) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.body = body


def _is_tty() -> bool:
    return sys.stdout.isatty()


def _color(text: str, code: str) -> str:
    if not _is_tty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _ok(msg: str) -> None:
    print(_color("[OK] ", "32") + msg)


def _err(msg: str) -> None:
    print(_color("[ERR] ", "31") + msg, file=sys.stderr)


def _info(msg: str) -> None:
    print(_color("[..] ", "36") + msg)


def _extract_message(body: Any, fallback: str) -> str:
    if isinstance(body, dict):
        for key in ("msg", "message", "error", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(body, str) and body.strip():
        return body.strip()[:400]
    return fallback


def _unwrap(body: Any) -> Any:
    if isinstance(body, dict) and "data" in body and (
        body.get("code") in (0, "0", None) or "msg" in body
    ):
        data = body.get("data")
        if data is not None:
            return data
    return body


def _login_type(account: str) -> str:
    account = account.strip()
    if re.fullmatch(r"1\d{10}", account) or re.fullmatch(r"\d{6,15}", account):
        return "phone"
    return "email"


def _strip_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_accounts_env(path: Path) -> dict[str, str]:
    slots = parse_account_slots(path)
    return dict(slots[0]) if slots else {}


def parse_account_slots(path: Path) -> list[dict[str, str]]:
    """Read email/code pairs; optional slot=N precedes each pair for stable aliases."""
    slots: list[dict[str, str]] = []
    current: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip().lstrip("\ufeff").lower()
        value = _strip_env_value(value)
        if not key:
            continue
        if key in ACCOUNT_KEYS:
            if current.get("account") and current.get("password"):
                slots.append(current)
                current = {}
            elif current.get("account"):
                current = {}
            current["account"] = value
        elif key in PASSWORD_KEYS:
            current["password"] = value
            if current.get("account"):
                slots.append(current)
                current = {}
        else:
            current[key] = value
    if current.get("account") or current.get("password"):
        slots.append(current)
    if any("slot" in entry for entry in slots):
        aliases = set()
        for entry in slots:
            alias = entry.get("slot", "")
            if not re.fullmatch(r"[1-9][0-9]*", alias) or alias in aliases:
                raise ValueError("显式 slot 必须是互不重复的正整数，且每组账号都须指定 slot")
            aliases.add(alias)
    return slots


def find_accounts_env(explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path if path.is_file() else path
    for candidate in DEFAULT_ENV_FILES:
        if candidate.is_file():
            return candidate
    return None


def load_credentials(
    *,
    account: str | None = None,
    env_file: str | None = None,
    prompt: bool = False,
    slot: int = 1,
) -> dict[str, str]:
    slots: list[dict[str, str]] = []
    source = ""
    path = find_accounts_env(env_file)
    if path is not None and path.is_file():
        slots = parse_account_slots(path)
        source = str(path)
    elif env_file:
        raise FileNotFoundError(f"找不到账号文件: {path}")

    if slot < 1:
        raise ValueError(f"slot 从 1 开始，收到 {slot}")
    if any("slot" in entry for entry in slots):
        # A sparse file may contain only the newly added account. Unlisted
        # aliases keep their cached sessions instead of inheriting its credentials.
        file_values = next((entry for entry in slots if entry["slot"] == str(slot)), {})
    else:
        file_values = slots[slot - 1] if slot <= len(slots) else {}
        if slots and slot > len(slots):
            raise ValueError(f"{path} 只有 {len(slots)} 个账号，没有 slot {slot}")

    file_account = file_values.get("account", "")
    file_password = file_values.get("password", "")

    resolved_account = (
        (account or "").strip()
        or os.environ.get("CANNJUDGE_ACCOUNT", "").strip()
        or file_account
    )
    resolved_password = os.environ.get("CANNJUDGE_PASSWORD", "").strip() or file_password

    if prompt:
        if not resolved_account:
            resolved_account = input("账号（邮箱/手机）: ").strip()
        if not resolved_password:
            resolved_password = getpass.getpass("密码: ")

    origin = source
    if source:
        origin = f"{source} slot {slot}" if file_values else ""
    return {
        "account": resolved_account,
        "password": resolved_password,
        "source": origin,
        "slot": str(slot),
    }


def _fmt_time(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    # The judge returns microseconds for every case, including the large ones.
    if number < 1000:
        return f"{number:.2f}μs"
    if number < 1000000:
        return f"{number / 1000:.2f}ms"
    return f"{number / 1000000:.2f}s"


def _status_style(status: str) -> str:
    lowered = status.lower()
    if status in {"Pass", "AC", "Accepted"} or lowered == "pass":
        return _color(status, "32")
    if status in {"WA", "TLE", "RE", "CE", "Fail", "Failed"} or "fail" in lowered:
        return _color(status, "31")
    return _color(status, "33")


class CannJudgeClient:
    def __init__(self, cache_dir: Path = CACHE_DIR, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cookie_path = self.cache_dir / "cookies.txt"
        self._user_path = self.cache_dir / "user.json"
        self._state_path = self.cache_dir / "state.json"
        self.cookies = http.cookiejar.LWPCookieJar(str(self._cookie_path))
        if self._cookie_path.exists():
            try:
                self.cookies.load(ignore_discard=True, ignore_expires=True)
            except (OSError, http.cookiejar.LoadError):
                pass
        self.user: dict[str, Any] | None = None
        if self._user_path.exists():
            try:
                self.user = json.loads(self._user_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.user = None
        ssl_ctx = ssl.create_default_context()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies),
            urllib.request.HTTPSHandler(context=ssl_ctx),
        )

    @property
    def logged_in(self) -> bool:
        return bool(self.user and self.user.get("_id"))

    @property
    def user_id(self) -> str:
        if not self.logged_in:
            raise ApiError(401, "未登录，请先运行: python3 scripts/cannjudge.py login")
        return str(self.user["_id"])

    def _save_session(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cookies.save(ignore_discard=True, ignore_expires=True)
        if self.user is not None:
            self._user_path.write_text(
                json.dumps(self.user, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            try:
                os.chmod(self._user_path, 0o600)
                os.chmod(self._cookie_path, 0o600)
            except OSError:
                pass

    def load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save_state(self, **updates: Any) -> dict[str, Any]:
        state = self.load_state()
        state.update({k: v for k, v in updates.items() if v is not None})
        self._state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return state

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            compact = {k: v for k, v in query.items() if v is not None and v != ""}
            if compact:
                url += "?" + urllib.parse.urlencode(compact)
        headers = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": USER_AGENT,
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/",
        }
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=60) as resp:
                raw = resp.read()
                if not raw:
                    return None
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            body: Any
            try:
                body = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = raw.decode("utf-8", errors="replace")
            raise ApiError(exc.code, _extract_message(body, exc.reason), body) from exc
        except urllib.error.URLError as exc:
            raise ApiError(0, f"网络错误: {exc.reason}") from exc

    def login(self, account: str, password: str) -> dict[str, Any]:
        body = self.request(
            "POST",
            "/api/users/login",
            json_body={
                "loginType": _login_type(account),
                "account": account.strip(),
                "password": password,
            },
        )
        user = _unwrap(body)
        if not isinstance(user, dict) or not user.get("_id"):
            raise ApiError(500, "登录成功但未返回用户信息")
        self.user = user
        self._save_session()
        return user

    def logout(self) -> None:
        self.user = None
        self.cookies.clear()
        for path in (self._cookie_path, self._user_path):
            if path.exists():
                path.unlink()

    def get_public_group(self) -> dict[str, Any]:
        group = _unwrap(self.request("GET", "/api/groups/public"))
        if not isinstance(group, dict) or not group.get("_id"):
            raise ApiError(500, "无法获取 public 小组")
        return group

    def get_contests(self, group_id: str) -> list[dict[str, Any]]:
        body = _unwrap(self.request("GET", f"/api/contests/group/{group_id}"))
        if not isinstance(body, list):
            raise ApiError(500, "竞赛列表格式异常")
        return body

    def get_contest_problems(self, contest_id: str) -> list[dict[str, Any]]:
        query = {"userId": self.user["_id"]} if self.logged_in else None
        body = _unwrap(
            self.request("GET", f"/api/problems/contest/{contest_id}", query=query)
        )
        if not isinstance(body, list):
            raise ApiError(500, "题目列表格式异常")
        return body

    def get_problem(self, problem_id: str) -> dict[str, Any]:
        query = {"userId": self.user["_id"]} if self.logged_in else None
        body = _unwrap(
            self.request("GET", f"/api/problems/{problem_id}", query=query)
        )
        if not isinstance(body, dict):
            raise ApiError(500, "题目详情格式异常")
        return body

    def get_template(self, problem_id: str) -> dict[str, Any]:
        query = {"userId": self.user["_id"]} if self.logged_in else None
        body = self.request(
            "GET", f"/api/problems/{problem_id}/template", query=query
        )
        data = _unwrap(body)
        if not isinstance(data, dict):
            raise ApiError(500, "模板格式异常")
        return data

    def resolve_problem(self, contest_name: str, problem_name: str) -> dict[str, Any]:
        group = self.get_public_group()
        contests = self.get_contests(group["_id"])
        contest = next(
            (c for c in contests if str(c.get("name", "")).lower() == contest_name.lower()),
            None,
        )
        if contest is None:
            raise ApiError(404, f"找不到赛事: {contest_name}")
        problems = self.get_contest_problems(contest["_id"])
        problem = next(
            (
                p
                for p in problems
                if str(p.get("name", "")).lower() == problem_name.lower()
            ),
            None,
        )
        if problem is None:
            raise ApiError(404, f"找不到题目: {problem_name}")
        problem = dict(problem)
        problem["contest_name"] = contest.get("name")
        problem["contest_id"] = contest["_id"]
        problem["contest_title"] = contest.get("title")
        return problem

    def submit(self, payload: dict[str, Any]) -> str:
        payload = dict(payload)
        payload["userId"] = self.user_id
        validate_request_size(payload)
        body = self.request("POST", "/api/submissions/submit", json_body=payload)
        data = _unwrap(body)
        if isinstance(data, dict) and data.get("submissionId"):
            return str(data["submissionId"])
        raise ApiError(500, f"提交成功但未返回 submissionId: {body!r}")

    def get_submission(self, submission_id: str) -> dict[str, Any]:
        body = _unwrap(
            self.request(
                "GET",
                f"/api/submissions/{submission_id}",
                query={"userId": self.user_id},
            )
        )
        if not isinstance(body, dict):
            raise ApiError(500, "提交详情格式异常")
        return body


def _accounts_match(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def _login_with_credentials(client: CannJudgeClient, creds: dict[str, str]) -> dict[str, Any]:
    user = client.login(creds["account"], creds["password"])
    nick = user.get("nickname") or user.get("email") or user["_id"]
    origin = creds.get("source") or "命令行/环境变量"
    _ok(f"登录成功  {nick}  id={user['_id']}")
    _info(f"账号来源: {origin}")
    _info(f"会话缓存在 {client.cache_dir}")
    return user


def _require_login(
    client: CannJudgeClient,
    env_file: str | None = None,
    slot: int = 1,
) -> None:
    creds = load_credentials(env_file=env_file, slot=slot, prompt=False)
    wanted = creds.get("account") or ""
    if client.logged_in:
        current = str((client.user or {}).get("email") or "")
        if not wanted:
            return
        if wanted and current and _accounts_match(current, wanted):
            return
        if wanted and current and not _accounts_match(current, wanted):
            _info(f"当前会话是 {current}，切换到 slot {slot} ({wanted})")
            client.logout()
    if creds["account"] and creds["password"]:
        _login_with_credentials(client, creds)
        return
    _err("未登录。请先放好 scripts/acounts.env，或执行: python3 scripts/cannjudge.py login")
    raise SystemExit(1)


def _include_basename(raw: str) -> str:
    return raw.replace("\\", "/").split("/")[-1]


def quoted_includes(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        match = INCLUDE_RE.match(stripped)
        if match:
            names.append(_include_basename(match.group(1)))
    return names


def kernel_units_cmake_path(src: Path) -> Path | None:
    for path in (src / "kernel_units.cmake",):
        if path.is_file():
            return path
    return None


def parse_cmake_kernel_units(cmake: Path) -> list[str]:
    text = cmake.read_text(encoding="utf-8")
    match = CMAKE_UNITS_RE.search(text)
    if not match:
        raise FileNotFoundError(f"{cmake} 里没有 SINKHORN_KERNEL_UNITS")
    names = CMAKE_UNIT_NAME_RE.findall(match.group(1))
    if "kernel.asc" not in names:
        raise FileNotFoundError(f"{cmake} 的 SINKHORN_KERNEL_UNITS 缺少 kernel.asc")
    return names


def local_include_names(src: Path, text: str) -> list[str]:
    """Quoted includes that exist next to the kernel and can be submitted."""
    names: list[str] = []
    for inc in quoted_includes(text):
        if inc in PROTECTED_FILES or inc in SYSTEM_INCLUDES:
            continue
        if not KERNEL_FILE_RE.fullmatch(inc):
            continue
        if (src / inc).is_file():
            names.append(inc)
    return names


def add_unit(selected: dict[str, Path], src: Path, name: str) -> None:
    if name in selected:
        return
    local = src / name
    if not local.is_file():
        raise FileNotFoundError(f"缺少提交文件: {local}")
    if not KERNEL_FILE_RE.fullmatch(name):
        raise FileNotFoundError(f"文件名不能提交（仅 .asc/.h）: {name}")
    selected[name] = local


def close_local_includes(selected: dict[str, Path], src: Path) -> None:
    pending = list(selected)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        text = selected[name].read_text(encoding="utf-8")
        for inc in local_include_names(src, text):
            if inc not in selected:
                add_unit(selected, src, inc)
                pending.append(inc)


def ensure_payload_complete(src: Path, files: list[dict[str, str]]) -> None:
    """Refuse to submit if cmake units or local #includes are missing from the zip."""
    src = src.resolve()
    by_path = {item["path"]: item["content"] for item in files}
    problems: list[str] = []
    for item in files:
        name = item["path"]
        if name in PROTECTED_FILES or not KERNEL_FILE_RE.fullmatch(name):
            problems.append(f"不允许提交只读文件或非平级 .asc/.h 文件: {name}")
    if "kernel.asc" not in by_path:
        problems.append("缺少 kernel.asc")

    cmake = kernel_units_cmake_path(src)
    if cmake is not None:
        for name in parse_cmake_kernel_units(cmake):
            if name not in by_path:
                problems.append(f"kernel_units.cmake 列出但未提交: {name}")

    for name, content in by_path.items():
        for inc in local_include_names(src, content):
            if inc not in by_path:
                problems.append(f'{name} #include "{inc}" 未提交')

    if problems:
        raise FileNotFoundError("拒绝提交，文件不完整:\n  " + "\n  ".join(problems))


def collect_files(src: Path, template: dict[str, Any] | None) -> list[dict[str, str]]:
    src = src.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"源码目录不存在: {src}")

    selected: dict[str, Path] = {}
    template_files = []
    if template:
        template_files = template.get("files") or []

    for item in template_files:
        path = str(item.get("path") or "").replace("\\", "/")
        if not path or item.get("editable") is False:
            continue
        if path in PROTECTED_FILES or not KERNEL_FILE_RE.fullmatch(path):
            raise ValueError(f"模板可编辑文件不受当前工具支持: {path}")
        local = src / path
        if not local.is_file():
            raise FileNotFoundError(f"模板要求提交 {path}，本地找不到: {local}")
        selected[path] = local

    add_unit(selected, src, "kernel.asc")

    cmake = kernel_units_cmake_path(src)
    cmake_units: list[str] = []
    if cmake is not None:
        cmake_units = parse_cmake_kernel_units(cmake)
        for name in cmake_units:
            add_unit(selected, src, name)

    readonly = {f.get("path") for f in template_files if f.get("editable") is False}
    for child in sorted(src.iterdir()):
        name = child.name
        if not child.is_file() or name in selected or name in PROTECTED_FILES or name in readonly:
            continue
        if KERNEL_FILE_RE.fullmatch(name):
            selected[name] = child

    close_local_includes(selected, src)

    extra = sorted(name for name in selected if cmake_units and name not in cmake_units)
    if extra:
        _info("cmake 未列出、按目录/#include 补上: " + ", ".join(extra))

    files = [{"path": path, "content": local.read_text(encoding="utf-8")} for path, local in selected.items()]
    ensure_payload_complete(src, files)
    return files


def build_payload(
    problem_id: str, files: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "problemId": problem_id,
        "files": files,
        "tiling_h": "",
        "tiling_key_h": "",
        "host_cpp": "",
        "kernel_cpp": "",
    }


def prepare_upload_files(files: list[dict[str, str]]):
    """Preserve the new problem's source verbatim; never prune old-topic branches."""
    if len(files) > 20:
        raise ValueError("提交文件超过本地 20 文件预算，请先整理源码")
    prepared = [dict(f) for f in files]
    audit = {
        "source_file_count": len(files),
        "upload_file_count": len(prepared),
        "upload_source_bytes": sum(len(f["content"].encode("utf-8")) for f in prepared),
        "hash_checks": False,
        "request_bytes": validate_upload_size(prepared),
        "request_budget_bytes": UPLOAD_REQUEST_BUDGET,
    }
    return prepared, audit


def validate_upload_size(files, problem_id="0" * 24, user_id="0" * 24):
    payload = dict(build_payload(problem_id, files), userId=user_id)
    return validate_request_size(payload)


def validate_request_size(payload):
    size = len(json.dumps(payload).encode("utf-8"))
    if size > UPLOAD_REQUEST_BUDGET:
        raise ValueError(f"拒绝上传：请求 {size:,} B 超过本地安全预算 {UPLOAD_REQUEST_BUDGET:,} B；"
                         "请先检查提交副本，避免再次触发413。")
    return size


def print_upload_audit(audit):
    _info("提交副本保持源码原文，不裁剪、不压缩")
    _info(f"提交源码 {audit['upload_source_bytes']:,} B；JSON请求约 {audit['request_bytes']:,} B"
          f" / 本地预算 {UPLOAD_REQUEST_BUDGET:,} B")


def cmd_prepare(args: argparse.Namespace) -> int:
    """Materialize exactly the prepared payload without credentials or network."""
    src = Path(args.src).resolve()
    dst = Path(args.out).resolve()
    if dst == src or dst in src.parents or src in dst.parents or dst.exists():
        raise ValueError("输出必须是源码目录以外的全新目录；不覆盖任何已有本地文件")
    files, audit = prepare_upload_files(collect_files(src, None))
    dst.mkdir(parents=True)
    for item in files:
        (dst / item["path"]).write_text(item["content"], encoding="utf-8")
    # A staging manifest must describe the retained files, not the original TU.
    (dst / "kernel_units.cmake").write_text("set(SINKHORN_KERNEL_UNITS\n" +
        "".join(f"    {f['path']}\n" for f in files) + ")\n", encoding="utf-8")
    (dst / "upload-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (dst / "uploaded-files.json").write_text(json.dumps(files, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_upload_audit(audit)
    _ok(f"已生成提交副本：{dst}；未联网、未提交")
    return 0


def print_file_summary(files: Iterable[dict[str, str]], src: Path | None = None) -> None:
    rows = list(files)
    required: set[str] = set()
    if src is not None:
        cmake = kernel_units_cmake_path(src)
        if cmake is not None:
            required = set(parse_cmake_kernel_units(cmake))
    _info(f"将提交 {len(rows)} 个文件：")
    for item in rows:
        n_lines = item["content"].count("\n") + (0 if item["content"].endswith("\n") else 1)
        tag = "必交" if (not required or item["path"] in required) else "额外"
        print(f"  [{tag}] {item['path']:<24} {len(item['content']):>8} bytes  {n_lines} lines")
    if required:
        present = {item["path"] for item in rows}
        for item in rows:
            present.update(re.findall(r'^#line 1 "\./([^"/]+\.asc)"$', item["content"], re.M))
        missing = sorted(required - present)
        if missing:
            raise FileNotFoundError("提交清单缺少必交文件: " + ", ".join(missing))


def print_submission(sub: dict[str, Any]) -> None:
    status = str(sub.get("status") or "?")
    score = sub.get("score")
    print(f"  状态: {_status_style(status)}    分数: {score}")
    for key in ("msg", "message", "error", "fail_reason", "compile_log"):
        value = sub.get(key)
        if isinstance(value, str) and value.strip():
            print(f"  {key}: {value.strip()[:2000]}")
    results = sub.get("result") or sub.get("results") or []
    if not isinstance(results, list) or not results:
        return
    print(f"  测试点: {len(results)}")
    for idx, case in enumerate(results, start=1):
        if not isinstance(case, dict):
            print(f"    #{idx} {case}")
            continue
        case_status = str(
            case.get("testcase_status")
            or case.get("status")
            or case.get("result")
            or "?"
        )
        elapsed = case.get("time")
        if elapsed is None:
            elapsed = case.get("runtime")
        best = case.get("best_time")
        line = f"    #{idx:<2} {_status_style(case_status):<12} {_fmt_time(elapsed)}"
        if best is not None:
            line += f"  best={_fmt_time(best)}"
        print(line)


def poll_until_done(
    client: CannJudgeClient, submission_id: str, interval: int, timeout: int
) -> dict[str, Any]:
    started = time.time()
    last_key = None
    while True:
        sub = client.get_submission(submission_id)
        status = str(sub.get("status") or "")
        results = sub.get("result") or []
        n_cases = len(results) if isinstance(results, list) else 0
        key = (status, sub.get("score"), n_cases)
        if key != last_key:
            last_key = key
            print(f"  [{_status_style(status)}] score={sub.get('score')}  ({n_cases} cases)")
            if n_cases:
                print_submission(sub)
        if status and status not in PENDING_STATUSES:
            return sub
        if time.time() - started > timeout:
            raise TimeoutError(f"等待评测超时（{timeout}s），提交 ID: {submission_id}")
        time.sleep(max(1, interval))


def cmd_login(args: argparse.Namespace) -> int:
    client = CannJudgeClient(cache_dir=CACHE_DIR / f"slot{getattr(args, 'slot', 1)}")
    creds = load_credentials(
        account=args.account,
        env_file=args.env,
        prompt=True,
        slot=getattr(args, "slot", 1),
    )
    if not creds["account"] or not creds["password"]:
        _err("账号或密码为空。请在 scripts/acounts.env 写入 email / code")
        return 1
    _login_with_credentials(client, creds)
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    CannJudgeClient(cache_dir=CACHE_DIR / f"slot{args.slot}").logout()
    _ok("已退出登录")
    return 0


def cmd_whoami(args: argparse.Namespace) -> int:
    client = CannJudgeClient(cache_dir=CACHE_DIR / f"slot{getattr(args, 'slot', 1)}")
    _require_login(client, args.env, slot=getattr(args, "slot", 1))
    user = client.user or {}
    print(f"  _id:      {user.get('_id')}")
    print(f"  nickname: {user.get('nickname')}")
    print(f"  email:    {user.get('email')}")
    print(f"  slot:     {getattr(args, 'slot', 1)}")
    return 0


def _prepare_submit(args: argparse.Namespace) -> tuple[CannJudgeClient, dict[str, Any], list[dict[str, str]], Path]:
    client = CannJudgeClient(cache_dir=CACHE_DIR / f"slot{getattr(args, 'slot', 1)}")
    if not args.dry_run:
        _require_login(client, getattr(args, "env", None), slot=getattr(args, "slot", 1))
    problem = client.resolve_problem(args.contest, args.problem)
    template = client.get_template(problem["_id"])
    src = Path(args.src) if args.src else DEFAULT_SRC
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    files = collect_files(src, template)
    if args.file:
        extra = {}
        for rel in args.file:
            path = Path(rel)
            local = path if path.is_absolute() else (src / path)
            if not local.is_file():
                raise FileNotFoundError(f"指定文件不存在: {local}")
            extra[local.name if path.name == rel else str(path)] = local
        merged = {item["path"]: item for item in files}
        for name, local in extra.items():
            merged[name] = {"path": name, "content": local.read_text(encoding="utf-8")}
        files = list(merged.values())
    ensure_payload_complete(src, files)
    readonly = {f.get("path") for f in template.get("files", []) if f.get("editable") is False}
    if any(f["path"] in readonly for f in files):
        raise ValueError("拒绝提交线上模板的只读文件")
    files, audit = prepare_upload_files(files)
    validate_upload_size(files, problem["_id"], client.user_id if client.logged_in else "0" * 24)
    print_upload_audit(audit)
    return client, problem, files, src


def cmd_info(args: argparse.Namespace) -> int:
    client = CannJudgeClient(cache_dir=CACHE_DIR / f"slot{getattr(args, 'slot', 1)}")
    problem = client.resolve_problem(args.contest, args.problem)
    print(f"  赛事: {problem.get('contest_title')}  ({problem.get('contest_name')})")
    print(f"  题目: {problem.get('title')}  ({problem.get('name')})")
    print(f"  contest_id: {problem.get('contest_id')}")
    print(f"  problem_id: {problem.get('_id')}")
    print(f"  code_template: {problem.get('code_template')}")
    template = client.get_template(problem["_id"])
    files = template.get("files") or []
    print(f"  模板文件 ({len(files)}):")
    for item in files:
        flag = "可提交" if item.get("editable") is not False else "只读"
        print(f"    [{flag}] {item.get('path')}")
    src = Path(args.src) if args.src else DEFAULT_SRC
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    local_files = collect_files(src, template)
    local_files, audit = prepare_upload_files(local_files)
    print_upload_audit(audit)
    # This is the exact payload; no source transformation is performed.
    print_file_summary(local_files)
    return 0


def committed_source_version(src: Path, files: list[dict[str, str]]) -> str:
    """Associate the exact upload text with a committed source tree, without hashes."""
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(src), *args],
                                capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            raise ValueError("真实提交要求源码位于 Git 工作区且所有上传源码已提交")
        return result.stdout

    root = Path(git("rev-parse", "--show-toplevel").strip())
    version = git("rev-parse", "HEAD").strip()
    for item in files:
        path = (src / item["path"]).resolve()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            raise ValueError(f"上传源码位于 Git 工作区以外: {path}") from None
        if git("show", f"{version}:{rel}") != item["content"]:
            raise ValueError(f"上传内容与 Git HEAD 不一致，请先提交源码: {rel}")
    return version


def cmd_submit(args: argparse.Namespace) -> int:
    client, problem, files, src = _prepare_submit(args)
    print_file_summary(files)
    payload = build_payload(problem["_id"], files)
    if args.dry_run:
        _ok("dry-run：未真正提交")
        return 0
    if any("TODO: define your launch kernel." in f["content"] for f in files):
        raise ValueError("当前仍是官方空实现模板。请先适配并验证 run_kernel，再真实提交")
    code_version = committed_source_version(src, files)
    _info(f"实际测试代码版本: {code_version}")
    submission_id = client.submit(payload)
    contest_name = problem.get("contest_name") or args.contest
    problem_name = problem.get("name") or args.problem
    url = f"{BASE_URL}/public/{contest_name}/{problem_name}/submission/{submission_id}"
    client.save_state(
        last_submission_id=submission_id,
        problem_id=problem["_id"],
        contest_id=problem.get("contest_id"),
        contest_name=contest_name,
        problem_name=problem_name,
        code_version=code_version,
    )
    _ok(f"已提交  submission_id={submission_id}")
    print(f"  {url}")
    if args.watch:
        _info("等待评测...")
        sub = poll_until_done(client, submission_id, args.interval, args.timeout)
        print()
        _ok("评测结束")
        print_submission(sub)
        status = str(sub.get("status") or "")
        return 0 if status in {"Pass", "AC", "Accepted"} else 2
    return 0


def cmd_result(args: argparse.Namespace) -> int:
    client = CannJudgeClient(cache_dir=CACHE_DIR / f"slot{getattr(args, 'slot', 1)}")
    _require_login(client, getattr(args, "env", None), slot=getattr(args, "slot", 1))
    submission_id = args.submission_id or client.load_state().get("last_submission_id")
    if not submission_id:
        _err("没有 submission id。先 submit，或手动传入 ID。")
        return 1
    if args.watch:
        _info(f"轮询 {submission_id} ...")
        sub = poll_until_done(client, submission_id, args.interval, args.timeout)
        print()
        _ok("评测结束")
        print_submission(sub)
        return 0
    sub = client.get_submission(submission_id)
    print(f"  submission_id: {submission_id}")
    print_submission(sub)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cannjudge",
        description="向 cannjudge.cn 提交 SinkhornNorm（只提交可编辑的 kernel 文件）",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_env_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--env",
            help="账号文件路径（默认 scripts/acounts.env 或 scripts/accounts.env）",
        )
        p.add_argument(
            "--slot",
            type=int,
            default=1,
            help="账号槽位（默认 1）；显式 slot=N 优先，否则按文件顺序从 1 编号",
        )

    login = sub.add_parser("login", help="登录并缓存 cookie")
    add_env_arg(login)
    login.add_argument("--account", help="覆盖账号文件里的邮箱/手机号")
    login.set_defaults(func=cmd_login)

    logout = sub.add_parser("logout", help="清除选定账号的本地登录缓存")
    add_env_arg(logout)
    logout.set_defaults(func=cmd_logout)
    whoami = sub.add_parser("whoami", help="显示当前登录用户")
    add_env_arg(whoami)
    whoami.set_defaults(func=cmd_whoami)

    def add_problem_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--contest", default=DEFAULT_CONTEST, help="赛事 name")
        p.add_argument("--problem", default=DEFAULT_PROBLEM, help="题目 name")
        p.add_argument(
            "--src",
            default=str(DEFAULT_SRC),
            help="本地工程目录（默认 submission/）",
        )

    info = sub.add_parser("info", help="查看题目 ID 与将要提交的文件")
    add_problem_args(info)
    add_env_arg(info)
    info.set_defaults(func=cmd_info)

    prepare = sub.add_parser("prepare", help="离线生成原文提交副本，不登录、不联网、不提交")
    prepare.add_argument("--src", default=str(DEFAULT_SRC), help="本地工程目录")
    prepare.add_argument("--out", required=True, help="全新的提交副本目录")
    prepare.set_defaults(func=cmd_prepare)

    submit = sub.add_parser("submit", help="提交 submission/ 下可编辑 .asc/.h")
    add_env_arg(submit)
    add_problem_args(submit)
    submit.add_argument(
        "--file",
        action="append",
        default=[],
        help="额外提交的文件，可重复。默认按 SINKHORN_KERNEL_UNITS 和 #include 收齐 submission 下 .asc/.h",
    )
    submit.add_argument("--watch", "-w", action="store_true", help="提交后轮询结果")
    submit.add_argument("--interval", "-i", type=int, default=5, help="轮询间隔秒")
    submit.add_argument("--timeout", type=int, default=600, help="最长等待秒")
    submit.add_argument("--dry-run", action="store_true", help="只列出文件，不提交")
    submit.set_defaults(func=cmd_submit)

    result = sub.add_parser("result", help="查看最近一次或指定提交的评测结果")
    add_env_arg(result)
    result.add_argument("submission_id", nargs="?", help="省略则用上次 submit 的 ID")
    result.add_argument("--watch", "-w", action="store_true", help="轮询直到评测结束")
    result.add_argument("--interval", "-i", type=int, default=5, help="轮询间隔秒")
    result.add_argument("--timeout", type=int, default=600, help="最长等待秒")
    result.set_defaults(func=cmd_result)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print()
        _err("已中断")
        return 130
    except (ApiError, FileNotFoundError, TimeoutError, ValueError) as exc:
        _err(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
