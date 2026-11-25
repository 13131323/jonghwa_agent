"""외부 API 없이 CLI의 에이전트 재사용·실패 복구·종료를 검증한다."""

import io
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from jonghwa_agent.main import run_cli


class CliTests(unittest.TestCase):
    def test_multiple_questions_reuse_one_agent(self):
        agent = object()
        with patch("builtins.input", side_effect=["", "첫 질문", "둘째 질문", "/exit"]), \
             patch("jonghwa_agent.chatbot.create_personal_agent", return_value=agent) as create, \
             patch("jonghwa_agent.chatbot.invoke_agent", return_value={}) as invoke, \
             patch("jonghwa_agent.chatbot.answer_text", return_value="답변"), redirect_stdout(io.StringIO()):
            self.assertEqual(run_cli(), 0)
        create.assert_called_once()
        self.assertEqual([call.args for call in invoke.call_args_list], [(agent, "첫 질문"), (agent, "둘째 질문")])

    def test_request_failure_keeps_input_loop_running(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["첫 질문", "다음 질문", "/quit"]), \
             patch("jonghwa_agent.chatbot.create_personal_agent"), \
             patch("jonghwa_agent.chatbot.invoke_agent", side_effect=[RuntimeError("private error detail"), {}]) as invoke, \
             patch("jonghwa_agent.chatbot.answer_text", return_value="정상 답변"), \
             redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(run_cli(), 0)
        self.assertEqual(invoke.call_count, 2)
        self.assertIn("정상 답변", output.getvalue())
        self.assertNotIn("private error detail", output.getvalue())

    def test_eof_and_ctrl_c_exit_cleanly(self):
        for stop in (EOFError(), KeyboardInterrupt()):
            with self.subTest(stop=type(stop).__name__), \
                 patch("builtins.input", side_effect=stop), \
                 patch("jonghwa_agent.chatbot.create_personal_agent"), redirect_stdout(io.StringIO()):
                self.assertEqual(run_cli(), 0)

    def test_initialization_error_returns_failure(self):
        with patch("jonghwa_agent.chatbot.create_personal_agent", side_effect=ValueError("private setting")), \
             patch("builtins.input") as read, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(run_cli(), 1)
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
