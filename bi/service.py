"""Run Bi at login.

    python -m bi.service install     # systemd user service (Linux) / LaunchAgent (macOS) / Startup folder (Windows)
    python -m bi.service uninstall
    python -m bi.service status
    python -m bi.service print       # show the file that would be installed
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .config import PROJECT_ROOT, detect_os

LABEL = "com.bi.assistant"


def python_exe(windowless: bool = False) -> str:
    exe = Path(sys.executable)
    if windowless and os.name == "nt":
        pyw = exe.with_name("pythonw.exe")
        if pyw.exists():
            return str(pyw)
    return str(exe)


def systemd_unit(root: Path = PROJECT_ROOT, py: str | None = None) -> str:
    return f"""[Unit]
Description=Bi voice assistant
After=graphical-session.target pipewire.service pulseaudio.service

[Service]
Type=simple
WorkingDirectory={root}
ExecStart={py or python_exe()} {root / 'main.py'}
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""


def launchd_plist(root: Path = PROJECT_ROOT, py: str | None = None) -> str:
    logs = root / "data" / "logs"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array><string>{py or python_exe()}</string><string>{root / 'main.py'}</string></array>
  <key>WorkingDirectory</key><string>{root}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>{logs / 'launchd.out.log'}</string>
  <key>StandardErrorPath</key><string>{logs / 'launchd.err.log'}</string>
</dict>
</plist>
"""


def windows_vbs(root: Path = PROJECT_ROOT, py: str | None = None) -> str:
    # A .vbs launcher in the Startup folder starts Bi hidden at login without admin rights.
    exe = py or python_exe(windowless=True)
    return (f'Set sh = CreateObject("WScript.Shell")\r\n'
            f'sh.CurrentDirectory = "{root}"\r\n'
            f'sh.Run """{exe}"" ""{root / "main.py"}""", 0, False\r\n')


def target_path(system: str, home: Path | None = None) -> Path:
    home = home or Path.home()
    if system == "linux":
        return home / ".config" / "systemd" / "user" / "bi.service"
    if system == "macos":
        return home / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    if system == "windows":
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "Bi.vbs"
    raise SystemExit(f"Unsupported OS: {system}")


def content_for(system: str) -> str:
    return {"linux": systemd_unit, "macos": launchd_plist, "windows": windows_vbs}[system]()


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=False)


def install(system: str, enable: bool = True, home: Path | None = None) -> Path:
    path = target_path(system, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    path.write_text(content_for(system), encoding="utf-8")
    print(f"wrote {path}")
    if enable:
        if system == "linux":
            _run(["systemctl", "--user", "daemon-reload"])
            _run(["systemctl", "--user", "enable", "--now", "bi.service"])
            print("logs: journalctl --user -u bi -f")
        elif system == "macos":
            _run(["launchctl", "unload", str(path)])
            _run(["launchctl", "load", "-w", str(path)])
            print("macOS asks for microphone permission for Python the first time; approve it in "
                  "System Settings > Privacy & Security > Microphone.")
        else:
            print("Bi will start hidden at your next login. Start it now by double-clicking the file above.")
    return path


def uninstall(system: str, home: Path | None = None) -> None:
    path = target_path(system, home)
    if system == "linux":
        _run(["systemctl", "--user", "disable", "--now", "bi.service"])
    elif system == "macos" and path.exists():
        _run(["launchctl", "unload", "-w", str(path)])
    if path.exists():
        path.unlink()
        print(f"removed {path}")
    else:
        print("not installed")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bi.service", description="Start Bi automatically at login")
    p.add_argument("action", choices=["install", "uninstall", "status", "print"])
    p.add_argument("--no-enable", action="store_true", help="write the file but don't start it")
    a = p.parse_args(argv)
    system = detect_os().system
    if a.action == "print":
        print(f"# {target_path(system)}\n{content_for(system)}")
    elif a.action == "install":
        install(system, enable=not a.no_enable)
    elif a.action == "uninstall":
        uninstall(system)
    else:
        path = target_path(system)
        print(f"{'installed' if path.exists() else 'not installed'}: {path}")
        if system == "linux" and path.exists():
            _run(["systemctl", "--user", "status", "bi.service", "--no-pager"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
