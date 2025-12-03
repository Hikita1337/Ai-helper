#!/bin/bash
set -e

# 1. Создать папку
mkdir -p tools/megacmd
cd tools/megacmd

# 2. Скачать DEB пакет MEGAcmd
MEGA_DEB_URL="https://mega.nz/linux/MEGAsync/xUbuntu_22.04/amd64/megacmd-xUbuntu_22.04_amd64.deb"
wget -O megacmd.deb "$MEGA_DEB_URL"

# 3. Распаковать DEB без установки
dpkg-deb -x megacmd.deb ./megacmd_extracted

# 4. Сделать бинарник исполняемым
chmod +x ./megacmd_extracted/usr/bin/mega

echo "MEGAcmd готов к использованию в tools/megacmd/megacmd_extracted/usr/bin/mega"