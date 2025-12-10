"""Helper to build a Windows .exe with PyInstaller while guarding against invalid paths.

Usage (from repo root):
    python scripts/build_windows.py --name auto_seg --entry app.py

This script sanitizes the output name, shortens dist/work paths, checks
writeability before running PyInstaller, and provides clear guidance when
Windows rejects the target path (e.g., invalid characters or network drives).
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

DEFAULT_NAME = "auto_seg"


def sanitize_name(name: str) -> str:
    # Allow alnum, dash, underscore, dot; strip trailing dots/spaces Windows dislikes
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip(" .")
    if not cleaned:
        cleaned = DEFAULT_NAME
    # Keep names reasonably short to avoid MAX_PATH issues on some setups
    return cleaned[:80]


def ensure_path_writeable(exe_path: Path) -> None:
    exe_path.parent.mkdir(parents=True, exist_ok=True)
    probe = exe_path.with_suffix(exe_path.suffix + ".writecheck")
    try:
        with open(probe, "wb") as fp:  # noqa: PTH123
            fp.write(b"")
    except OSError as exc:  # noqa: B904
        raise SystemExit(
            "❌ 출력 경로에 쓸 수 없습니다. dist/work 경로에 특수문자, 공백, 네트워크 드라이브 여부를 확인하세요.\n"
            f"   경로: {exe_path}\n"
            f"   오류: {exc}"
        )
    else:
        probe.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Windows exe with PyInstaller (path-safe)")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Output executable base name (without .exe)")
    parser.add_argument("--entry", default="app.py", help="Entry point script for PyInstaller")
    parser.add_argument("--dist", default="dist", help="Destination folder for the exe")
    parser.add_argument("--build", default="build", help="PyInstaller workpath")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Additional PyInstaller args")
    args = parser.parse_args()

    exe_name = sanitize_name(args.name)
    dist_dir = Path(args.dist).resolve()
    work_dir = Path(args.build).resolve()
    entry = Path(args.entry).resolve()

    exe_path = dist_dir / f"{exe_name}.exe"
    ensure_path_writeable(exe_path)

    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--name",
        exe_name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        str(entry),
    ]
    if args.extra:
        cmd.extend(args.extra)

    print("[info] 실행 명령:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:  # noqa: B904
        raise SystemExit(
            "❌ PyInstaller 빌드 실패. 위 명령 출력에서 오류를 확인하세요.\n"
            "   dist/work 경로나 exe 이름에 윈도우에서 허용되지 않는 문자가 없는지 확인하거나\n"
            "   더 짧은 이름(--name)과 로컬 디스크 경로(--dist, --build)를 사용해 보세요."
        ) from exc


if __name__ == "__main__":
    main()
