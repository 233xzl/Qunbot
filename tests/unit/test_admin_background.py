"""Exercise the deployed CLI against an isolated instance and a real listener."""

import socket

import httpx

from group_agent.admin.cli import main


def test_background_restart_restores_engine_and_preserves_instance(tmp_path, capsys):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def command(action, *options):
        assert main([action, "--instance", str(tmp_path), *options]) == 0

    with httpx.Client(
        base_url=f"http://127.0.0.1:{port}", headers={"X-Qunbot-Request": "1"}, trust_env=False
    ) as client:

        def check(state):
            client.post("/api/login", json={"key": key}).raise_for_status()
            status = client.get("/api/status").raise_for_status().json()
            assert status["demo"] and status["state"] == state

        try:
            command("deploy", "--demo", "--port", str(port))
            key = capsys.readouterr().out.splitlines()[0].split("：", 1)[1]
            command("deploy")
            assert "管理密钥" not in capsys.readouterr().out
            check("stopped")
            command("start")
            client.post("/api/runtime/start", json={}).raise_for_status()
            command("restart")
            check("running")
            client.post("/api/runtime/pause", json={}).raise_for_status()
            command("restart")
            check("paused")
            client.post("/api/runtime/stop", json={}).raise_for_status()
            command("restart")
            check("stopped")
        finally:
            command("stop")
