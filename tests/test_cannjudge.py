"""Submission boundaries: exact source, read-only files, Git tracking and dry-run."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cannjudge", ROOT / "scripts/cannjudge.py")
cj = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cj
spec.loader.exec_module(cj)


class SubmissionTest(unittest.TestCase):
    def test_explicit_third_slot_keeps_old_cached_accounts(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(cj.os.environ, {}, clear=True):
            path = Path(temp) / "accounts.env"
            path.write_text("slot=3\nemail=new@example.invalid\ncode=secret\n")
            for slot in (1, 2):
                creds = cj.load_credentials(env_file=str(path), slot=slot)
                self.assertEqual(creds["account"], "")
                client = Mock(logged_in=True, user={"email": "cached@example.invalid"})
                cj._require_login(client, str(path), slot=slot)
                client.login.assert_not_called()
                client.logout.assert_not_called()
            creds = cj.load_credentials(env_file=str(path), slot=3)
            self.assertEqual(creds["account"], "new@example.invalid")
            self.assertEqual(creds["password"], "secret")

    def test_slot_mapping_rejects_ambiguity_and_preserves_legacy_order(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(cj.os.environ, {}, clear=True):
            path = Path(temp) / "accounts.env"
            pair = "email=test@example.invalid\ncode=secret\n"
            for content in ("slot=3\n" + pair + "slot=3\n" + pair,
                            "slot=3\n" + pair + pair, "slot=0\n" + pair):
                path.write_text(content)
                with self.assertRaisesRegex(ValueError, "slot"):
                    cj.load_credentials(env_file=str(path), slot=1)
            path.write_text(pair + "email=second@example.invalid\ncode=second\n")
            self.assertEqual(cj.load_credentials(env_file=str(path), slot=2)["password"], "second")

    def test_cached_login_works_without_credentials_file(self):
        client = Mock(logged_in=True, user={"email": "cached@example.invalid"})
        with patch.object(cj, "load_credentials", return_value={"account": "", "password": ""}):
            cj._require_login(client)
        client.login.assert_not_called()

    def test_uploads_only_kernel_verbatim(self):
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp)
            content = '// 注释保持原文\nint x;  /* spacing */\n'
            (src / "kernel.asc").write_text(content)
            (src / "main.asc").write_text("int main() {}\n")
            files = cj.collect_files(src, None)
            self.assertEqual([f["path"] for f in files], ["kernel.asc"])
            with patch("hashlib.sha256", side_effect=AssertionError("no routine hashes")):
                prepared, audit = cj.prepare_upload_files(files)
            self.assertEqual(prepared, files)
            self.assertEqual(files[0]["content"], content)
            self.assertFalse(audit["hash_checks"])

    def test_readonly_and_traversal_files_rejected(self):
        for name in ("main.asc", "reference.py", "../kernel.asc", "/tmp/kernel.asc"):
            with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                cj.ensure_payload_complete(ROOT / "submission", [
                    {"path": "kernel.asc", "content": "int x;\n"},
                    {"path": name, "content": "int y;\n"},
                ])

    def test_template_readonly_header_not_collected(self):
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp)
            (src / "kernel.asc").write_text("int x;\n")
            (src / "judge_data.h").write_text("int y;\n")
            template = {"files": [{"path": "kernel.asc", "editable": True},
                                  {"path": "judge_data.h", "editable": False}]}
            self.assertEqual([f["path"] for f in cj.collect_files(src, template)], ["kernel.asc"])

    def test_file_and_request_budgets(self):
        with self.assertRaises(ValueError):
            cj.prepare_upload_files([{"path": f"u{i}.asc", "content": ""} for i in range(21)])
        with self.assertRaises(ValueError):
            cj.prepare_upload_files([{"path": "kernel.asc", "content": "x" * cj.UPLOAD_REQUEST_BUDGET}])

    def test_dry_run_never_posts_submission(self):
        args = cj.build_parser().parse_args(["submit", "--dry-run"])
        client = Mock()
        files = [{"path": "kernel.asc", "content": "// TODO: define your launch kernel.\n"}]
        with patch.object(cj, "_prepare_submit", return_value=(client, {"_id": "problem"}, files, ROOT)):
            self.assertEqual(cj.cmd_submit(args), 0)
        client.submit.assert_not_called()

    def test_official_stub_cannot_be_submitted(self):
        args = cj.build_parser().parse_args(["submit"])
        client = Mock()
        files = [{"path": "kernel.asc", "content": "// TODO: define your launch kernel.\n"}]
        with patch.object(cj, "_prepare_submit", return_value=(client, {"_id": "problem"}, files, ROOT)):
            with self.assertRaisesRegex(ValueError, "官方空实现"):
                cj.cmd_submit(args)
        client.submit.assert_not_called()

    def test_source_version_requires_exact_committed_content(self):
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp)
            def git(*args):
                return subprocess.check_output(["git", "-C", temp, *args], text=True).strip()
            git("init", "-q")
            (src / "kernel.asc").write_text("int x;\n")
            git("add", "kernel.asc")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "core.hooksPath=/dev/null", "commit", "-qm", "source")
            files = [{"path": "kernel.asc", "content": "int x;\n"}]
            self.assertEqual(cj.committed_source_version(src, files), git("rev-parse", "HEAD"))
            files[0]["content"] = "int y;\n"
            with self.assertRaisesRegex(ValueError, "Git HEAD"):
                cj.committed_source_version(src, files)
            with self.assertRaises(ValueError):
                cj.committed_source_version(src, [{"path": "untracked.asc", "content": ""}])

    def test_logout_uses_selected_slot(self):
        args = cj.build_parser().parse_args(["logout", "--slot", "2"])
        with patch.object(cj, "CannJudgeClient") as client:
            cj.cmd_logout(args)
        client.assert_called_once_with(cache_dir=cj.CACHE_DIR / "slot2")
        client.return_value.logout.assert_called_once()


if __name__ == "__main__":
    unittest.main()
