import pytest

from agy_proxy.config import ConfigError, Settings


def test_defaults_generate_token_on_loopback():
    s = Settings.from_env({})
    assert s.proxy_token and s.token_generated
    assert s.host == "127.0.0.1" and s.enable_tool_trace is True
    assert s.proxy_token not in repr(s)


def test_secrets_not_in_repr():
    s = Settings(api_key="AIza-secret", proxy_token="tok-secret")
    assert "AIza-secret" not in repr(s) and "tok-secret" not in repr(s)


def test_explicit_token_not_generated():
    s = Settings.from_env({"AGY_PROXY_TOKEN": "abc"})
    assert s.proxy_token == "abc" and not s.token_generated


def test_non_loopback_without_token_has_no_generated_token():
    s = Settings.from_env({"AGY_HOST": "0.0.0.0"})
    assert s.proxy_token is None and not s.token_generated


def test_workspace_must_be_absolute_existing(tmp_path):
    assert Settings.from_env({"AGY_WORKSPACE": str(tmp_path)}).workspace == str(tmp_path.resolve())
    with pytest.raises(ConfigError, match="AGY_WORKSPACE"):
        Settings.from_env({"AGY_WORKSPACE": "relative/dir"})
    with pytest.raises(ConfigError, match="AGY_WORKSPACE"):
        Settings.from_env({"AGY_WORKSPACE": str(tmp_path / "missing")})


@pytest.mark.parametrize(
    "env, var",
    [
        ({"AGY_PORT": "abc"}, "AGY_PORT"),
        ({"AGY_PORT": "0"}, "AGY_PORT"),
        ({"AGY_SESSION_TTL": "x"}, "AGY_SESSION_TTL"),
        ({"AGY_SESSION_MAX": "0"}, "AGY_SESSION_MAX"),
        ({"AGY_LOG_TOOLS": "maybe"}, "AGY_LOG_TOOLS"),
        ({"AGY_ENABLE_CODER": "2"}, "AGY_ENABLE_CODER"),
        ({"AGY_ALLOWED_HOSTS": "http://evil/x"}, "AGY_ALLOWED_HOSTS"),
    ],
)
def test_bad_values_name_the_variable(env, var):
    with pytest.raises(ConfigError, match=var):
        Settings.from_env(env)


def test_flags_and_hosts():
    s = Settings.from_env(
        {"AGY_ENABLE_CODER": "1", "AGY_LOG_TOOLS": "false", "AGY_ALLOWED_HOSTS": "A.example, b"}
    )
    assert s.enable_coder and not s.enable_tool_trace
    assert s.allowed_hosts == ("a.example", "b")
    assert "127.0.0.1" in Settings.from_env({}).allowed_hosts
