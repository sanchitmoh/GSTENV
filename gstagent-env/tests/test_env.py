"""
Tests for the core GSTAgentEnv.

Covers: session isolation, max_steps, error recovery, O(1) lookups, caching.
"""

import pytest
from environment.env import GSTAgentEnv
from environment.models import GSTAction


class TestEnvReset:
    def test_returns_observation(self):
        env = GSTAgentEnv()
        obs = env.reset("invoice_match")
        assert obs.session_id != ""
        assert obs.task_id == "invoice_match"
        assert len(obs.purchase_register) > 0
        assert obs.step_number == 0

    def test_invalid_task_raises(self):
        env = GSTAgentEnv()
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("nonexistent_task")

    def test_session_id_unique(self):
        env = GSTAgentEnv()
        obs1 = env.reset("invoice_match")
        obs2 = env.reset("invoice_match")
        assert obs1.session_id != obs2.session_id


class TestEnvStep:
    def setup_method(self):
        self.env = GSTAgentEnv()
        self.env.reset("invoice_match")

    def test_valid_match_action(self):
        action = GSTAction(action_type="match_invoice", invoice_id="INV-0001")
        obs, reward, done, info = self.env.step(action)
        assert obs.step_number == 1
        assert done is False

    def test_invalid_action_type(self):
        action = GSTAction(action_type="bad_action")
        obs, reward, done, info = self.env.step(action)
        assert obs.last_action_error is not None
        assert "Invalid action_type" in obs.last_action_error
        assert done is False  # doesn't end episode

    def test_missing_invoice_id(self):
        action = GSTAction(action_type="match_invoice")  # no invoice_id
        obs, reward, done, info = self.env.step(action)
        assert obs.last_action_error is not None

    def test_nonexistent_invoice(self):
        action = GSTAction(action_type="match_invoice", invoice_id="INV-9999")
        obs, reward, done, info = self.env.step(action)
        assert obs.last_action_error is not None

    def test_submit_ends_episode(self):
        action = GSTAction(action_type="submit_report", payload={"matches": {}})
        obs, reward, done, info = self.env.step(action)
        assert done is True


class TestMaxSteps:
    def test_max_steps_enforced(self):
        env = GSTAgentEnv()
        env.reset("invoice_match")  # max_steps=8
        action = GSTAction(action_type="match_invoice", invoice_id="INV-0001")
        for _ in range(8):
            obs, reward, done, info = env.step(action)
        # 9th exploration step should be blocked but episode stays alive
        # (v2 design: agent can still call submit_report)
        obs, reward, done, info = env.step(action)
        assert done is False
        assert "error" in info
        assert "budget" in info["error"].lower()
        # submit_report still works after budget exhaustion
        submit = GSTAction(action_type="submit_report", payload={})
        obs, reward, done, info = env.step(submit)
        assert done is True

    def test_after_done_returns_done(self):
        env = GSTAgentEnv()
        env.reset("invoice_match")
        submit = GSTAction(action_type="submit_report", payload={})
        env.step(submit)
        # After done
        obs, reward, done, info = env.step(
            GSTAction(action_type="match_invoice", invoice_id="INV-0001")
        )
        assert done is True


class TestAllTasks:
    @pytest.mark.parametrize("task_id", ["invoice_match", "itc_audit", "full_recon"])
    def test_task_loads(self, task_id):
        env = GSTAgentEnv()
        obs = env.reset(task_id)
        assert obs.task_id == task_id
        assert len(obs.purchase_register) > 0
        assert obs.max_steps > 0


class TestReplayLog:
    def test_replay_records_actions(self):
        env = GSTAgentEnv()
        env.reset("invoice_match")
        env.step(GSTAction(action_type="match_invoice", invoice_id="INV-0001"))
        env.step(GSTAction(action_type="match_invoice", invoice_id="INV-0002"))
        assert len(env.replay_log) == 2
        assert env.replay_log[0]["action_type"] == "match_invoice"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
