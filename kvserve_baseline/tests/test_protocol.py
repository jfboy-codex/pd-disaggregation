import json
import sys
import unittest
from pathlib import Path


BASELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE))

from protocol import phase_payloads  # noqa: E402
from start_pd import backend_command, kv_config, load_config  # noqa: E402


class PhasePayloadTests(unittest.TestCase):
    def test_transfer_id_and_streaming(self):
        original = {
            "model": "qwen2.5-7b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 2048,
            "kv_transfer_params": {"transfer_id": "client-supplied"},
        }
        prefill, decode = phase_payloads(original, "stable-id")
        self.assertEqual(prefill["max_tokens"], 1)
        self.assertFalse(prefill["stream"])
        self.assertNotIn("stream_options", prefill)
        self.assertTrue(decode["stream"])
        self.assertEqual(decode["max_tokens"], 2048)
        self.assertEqual(decode["stream_options"], original["stream_options"])
        self.assertEqual(prefill["kv_transfer_params"]["transfer_id"], "stable-id")
        self.assertEqual(decode["kv_transfer_params"]["transfer_id"], "stable-id")
        self.assertTrue(prefill["kv_transfer_params"]["do_remote_decode"])
        self.assertTrue(decode["kv_transfer_params"]["do_remote_prefill"])
        self.assertEqual(original["kv_transfer_params"]["transfer_id"], "client-supplied")

    def test_reject_non_object(self):
        with self.assertRaises(ValueError):
            phase_payloads([], "id")


class LauncherConfigTests(unittest.TestCase):
    def test_ports_and_connector_roles(self):
        config = load_config(BASELINE / "config.json")
        producer = json.loads(kv_config(config, True))
        consumer = json.loads(kv_config(config, False))
        self.assertEqual(producer["kv_role"], "kv_producer")
        self.assertEqual(consumer["kv_role"], "kv_consumer")
        self.assertEqual(producer["kv_port"], consumer["kv_port"])
        self.assertEqual(producer["kv_connector_extra_config"], consumer["kv_connector_extra_config"])

        command = backend_command(config, Path("/model"), Path("/venv/bin/vllm"), True)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertIn("--no-enable-chunked-prefill", command)
        self.assertEqual(command[command.index("--port") + 1], "8100")


if __name__ == "__main__":
    unittest.main()
