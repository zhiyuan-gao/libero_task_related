from openpi.serving import websocket_policy_server


def test_keepalive_can_be_disabled_for_long_first_compile() -> None:
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=object(),
        ping_interval=None,
    )

    assert server._ping_interval is None  # noqa: SLF001
    assert server._ping_timeout == 20  # noqa: SLF001
