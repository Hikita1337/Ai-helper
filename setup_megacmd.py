#!/usr/bin/env python3
import os
import subprocess
import sys
import pathlib

MEGACMD_REPO = "https://github.com/meganz/MEGAcmd.git"
INSTALL_DIR = pathlib.Path.home() / "megacmd_build"
BACKUP_DIR = pathlib.Path.home() / "mega_backups"

def run(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout.decode())
    return result

def ensure_dirs():
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def install_dependencies():
    # для Ubuntu/Debian
    run(["sudo", "apt-get", "update"])
    run(["sudo", "apt-get", "install", "-y", "git", "cmake", "g++", "libssl-dev", "libsqlite3-dev", "make"])

def clone_megacmd():
    if not (INSTALL_DIR / "MEGAcmd").exists():
        run(["git", "clone", MEGACMD_REPO], cwd=INSTALL_DIR)

def build_megacmd():
    build_path = INSTALL_DIR / "MEGAcmd" / "build"
    build_path.mkdir(exist_ok=True)
    run(["cmake", ".."], cwd=build_path)
    run(["make"], cwd=build_path)
    print("MEGAcmd built successfully!")

def test_megacmd():
    megacmd_bin = INSTALL_DIR / "MEGAcmd" / "build" / "megacmd"
    if not megacmd_bin.exists():
        raise RuntimeError("MEGAcmd binary not found after build")
    run([str(megacmd_bin), "--help"])
    print("MEGAcmd is ready to use!")

def main():
    ensure_dirs()
    install_dependencies()
    clone_megacmd()
    build_megacmd()
    test_megacmd()
    print(f"All done! Backups can be stored in {BACKUP_DIR}")

if __name__ == "__main__":
    main()
