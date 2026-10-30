"""决策 4：champion / challenger / archived 治理层。

对齐 MLflow Model Registry 的 alias 机制：
promotion = 重新指向别名；rollback = 把别名指回旧版本（秒级，不是重新部署）；
旧版本 archived **不删除**，紧急回滚从 archived 取最近版本。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.governance import release as rel  # noqa: E402


class TestAliases(unittest.TestCase):
    """注意：测试跑在 `tests/_sandbox.py` 建立的临时 artifacts 副本里，
    改 ACTIVE 不会碰到仓库里的真实状态。

    ALIASES_FILE 额外指到临时目录：沙箱会**复制**真实的 ALIASES.json，
    若不换掉，ensure_aliases() 会读到真实 champion（rel-cand-34bfab62fe04），
    而 init_baseline() 把 ACTIVE 写成 baseline-v0 —— 两者不一致，测试会假失败。
    """

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.mkdtemp(prefix="rsi-alias-test-")
        self._saved = rel.ALIASES_FILE
        rel.ALIASES_FILE = Path(self._tmp) / "ALIASES.json"
        rel.init_baseline()

    def tearDown(self) -> None:
        rel.ALIASES_FILE = self._saved

    @staticmethod
    def _passing_gate(champion="baseline-v0", challenger="baseline-v0"):
        return {"promotable": True,
                "champion_release_id": champion,
                "challenger_release_id": challenger,
                "checks": [
            {"check": "protocol_same_origin", "passed": True},
            {"check": "main_dataset_results", "passed": True},
            {"check": "dataset:fixture", "passed": True,
             "detail": {"role": "main"}},
            {"check": "secondary_tolerances", "passed": True},
        ]}

    def test_champion_follows_active_on_first_adoption(self) -> None:
        """champion = 当前 ACTIVE；其余历史版本全部 archived 保留（不删）。"""
        data = rel.ensure_aliases(actor="test")
        self.assertEqual(data["champion"], rel.current_release_id())
        self.assertIsNone(data["challenger"])
        self.assertNotIn(data["champion"], data["archived"])
        self.assertTrue(len(data["archived"]) > 0)

    def test_set_challenger_does_not_change_active(self) -> None:
        before = rel.current_release_id()
        rel.set_alias("challenger", "baseline-v0", actor="test", reason="验证")
        self.assertEqual(rel.get_alias("challenger"), "baseline-v0")
        self.assertEqual(rel.current_release_id(), before)   # 登记≠生效

    def test_promote_requires_gate_and_signoff(self) -> None:
        rel.ensure_aliases(actor="test")
        rel.set_alias("challenger", "baseline-v0", actor="test", reason="验证")
        # 门没过 → 拒绝
        with self.assertRaises(rel.ReleaseError) as ctx:
            rel.promote_challenger(reviewer="pi", reason="x",
                                   promotion={"promotable": False, "blocking": ["主判据回退"]})
        self.assertIn("发布门未通过", str(ctx.exception))
        # 缺 sign-off → 拒绝
        with self.assertRaises(rel.ReleaseError):
            rel.promote_challenger(reviewer="", reason="x")
        with self.assertRaisesRegex(rel.ReleaseError, "不能跳过 gate"):
            rel.promote_challenger(reviewer="pi", reason="x")

    def test_promote_rejects_partial_or_inconsistent_passing_report(self) -> None:
        rel.ensure_aliases(actor="test")
        rel.set_alias("challenger", "baseline-v0", actor="test", reason="验证")
        for report in (
            {"promotable": True},
            {"promotable": True, "checks": [
                {"check": "protocol_same_origin", "passed": False},
                {"check": "main_dataset_results", "passed": True},
                {"check": "dataset:fixture", "passed": True,
                 "detail": {"role": "main"}},
                {"check": "secondary_tolerances", "passed": True},
            ]},
            {"promotable": True, "checks": [
                *TestAliases._passing_gate()["checks"],
                {"check": "secondary_tolerances", "passed": True},
            ]},
        ):
            with self.subTest(report=report), self.assertRaises(rel.ReleaseError):
                rel.promote_challenger(reviewer="pi", reason="x", promotion=report)

    def test_promote_binds_gate_to_current_champion_and_challenger(self) -> None:
        rel.ensure_aliases(actor="test")
        rel.set_alias("challenger", "baseline-v0", actor="test", reason="验证")
        with self.assertRaisesRegex(rel.ReleaseError, "challenger 与当前 alias"):
            rel.promote_challenger(
                reviewer="pi", reason="x",
                promotion=self._passing_gate(challenger="rel-other"))
        with self.assertRaisesRegex(rel.ReleaseError, "champion 已变化"):
            rel.promote_challenger(
                reviewer="pi", reason="x",
                promotion=self._passing_gate(champion="rel-old"))

    def test_promote_archives_previous_champion(self) -> None:
        """晋升 = 改指针：旧 champion 进 archived（不删除），ACTIVE 跟随 champion。"""
        rel.ensure_aliases(actor="test")
        prev = next(r for r in rel.list_releases() if r != "baseline-v0")
        rel.set_alias("champion", prev, actor="test", reason="构造晋升场景")
        rel.set_alias("challenger", "baseline-v0", actor="test", reason="验证")
        out = rel.promote_challenger(reviewer="pi", reason="门已过",
                                     promotion=self._passing_gate(
                                         champion=prev, challenger="baseline-v0"))
        self.assertEqual(out["to"], "baseline-v0")
        self.assertEqual(rel.current_release_id(), "baseline-v0")   # ACTIVE 跟随 champion
        self.assertIn(prev, rel.load_aliases()["archived"])          # 旧 champion 入 archived
        self.assertIsNone(rel.get_alias("challenger"))

    def test_rollback_syncs_champion(self) -> None:
        rel.ensure_aliases(actor="test")
        rel.rollback("baseline-v0")
        self.assertEqual(rel.get_alias("champion"), "baseline-v0")
        self.assertEqual(rel.current_release_id(), "baseline-v0")

    def test_rollback_surfaces_alias_sync_failure_after_active_recovery(self) -> None:
        from unittest.mock import patch

        rel.ensure_aliases(actor="test")
        other_release = next(value for value in rel.list_releases()
                             if value != "baseline-v0")
        rel.write_active(other_release)
        with patch.object(rel, "_write_aliases", side_effect=OSError("disk full")):
            result = rel.rollback("baseline-v0")

        self.assertEqual(rel.current_release_id(), "baseline-v0")
        self.assertEqual(result["governance_sync"]["status"], "failed")
        self.assertTrue(result["governance_sync"]["active_pointer_rolled_back"])
        self.assertEqual(result["governance_sync"]["error_type"], "OSError")

    def test_active_pointer_replace_failure_preserves_previous_release(self) -> None:
        from unittest.mock import patch

        before = rel.current_release_id()
        with patch.object(rel.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                rel.write_active("baseline-v0")

        self.assertEqual(rel.current_release_id(), before)
        self.assertEqual(list(rel.RELEASES_ROOT.glob("ACTIVE.*.tmp")), [])

    def test_rollback_cli_returns_failure_when_alias_sync_is_partial(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts import rsi

        with patch.object(rsi.rel, "current_release_id",
                          side_effect=["candidate", "baseline-v0"]), \
                patch.object(rsi.rel, "rollback", return_value={
                    "governance_sync": {"status": "failed"}}), \
                patch.object(rsi, "_print"):
            status = rsi.cmd_rollback(SimpleNamespace(to="baseline-v0", git=False))
        self.assertEqual(status, 1)

    def test_caveat_lifecycle_open_then_resolved(self) -> None:
        """caveat 挂上后必须能闭环：否则待办会变成永久噪声，真问题被淹没。"""
        rel.ensure_aliases(actor="test")
        rel.note_champion("X 问题待处理", actor="test")
        rel.note_champion("Y 问题待处理", actor="test")
        notes = rel.load_aliases()["champion_notes"]
        self.assertEqual([n.get("resolved") for n in notes], [None, None])

        out = rel.resolve_caveat("X 问题", actor="pi", reason="已修复")
        self.assertEqual(out["resolved"][0], 1)
        self.assertEqual(out["open"], [2])
        # 已解决的条目**保留**（审计留痕），只是打上 resolved
        self.assertTrue(rel.load_aliases()["champion_notes"][0].get("resolved"))

    def test_resolve_caveat_rejects_unknown(self) -> None:
        rel.ensure_aliases(actor="test")
        rel.note_champion("唯一 caveat", actor="test")
        with self.assertRaises(rel.ReleaseError):
            rel.resolve_caveat("不存在的条目", actor="test")
        # 幂等：已解决的不会被再次匹配
        rel.resolve_caveat("唯一 caveat", actor="test", reason="x")
        with self.assertRaises(rel.ReleaseError):
            rel.resolve_caveat("唯一 caveat", actor="test")

    def test_demote_keeps_release_available(self) -> None:
        """降级不是删除：版本仍在 list_releases 里，随时可回滚。"""
        rel.ensure_aliases(actor="test")
        champ = rel.get_alias("champion")
        rel.demote_champion(reviewer="pi", reason="复核发现假通过")
        self.assertEqual(rel.get_alias("challenger"), champ)
        self.assertIn(champ, rel.list_releases())


if __name__ == "__main__":
    unittest.main()
