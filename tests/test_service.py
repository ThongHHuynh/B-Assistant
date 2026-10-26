from pathlib import Path

from bi import service


def test_unit_files_reference_project_and_python(tmp_path):
    root, py = Path("/opt/bi"), "/opt/bi/.venv/bin/python"
    unit = service.systemd_unit(root, py)
    assert f"ExecStart={py} /opt/bi/main.py" in unit and "WorkingDirectory=/opt/bi" in unit
    assert "WantedBy=default.target" in unit
    plist = service.launchd_plist(root, py)
    assert "<string>com.bi.assistant</string>" in plist and f"<string>{py}</string>" in plist
    assert "<key>RunAtLoad</key><true/>" in plist
    vbs = service.windows_vbs(Path("C:/bi"), "C:/bi/.venv/Scripts/pythonw.exe")
    assert 'sh.Run """C:/bi/.venv/Scripts/pythonw.exe""' in vbs and ", 0, False" in vbs   # 0 = hidden window


def test_install_and_uninstall_writes_to_user_locations(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setattr(service, "_run", lambda cmd: None)
    for system, rel in [("linux", ".config/systemd/user/bi.service"),
                        ("macos", "Library/LaunchAgents/com.bi.assistant.plist")]:
        path = service.install(system, enable=False, home=tmp_path)
        assert path == tmp_path / rel and path.exists()
        service.uninstall(system, home=tmp_path)
        assert not path.exists()
    win = service.install("windows", enable=False, home=tmp_path)
    assert win.name == "Bi.vbs" and "Startup" in str(win)
