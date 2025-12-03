#!/bin/bash
set -e

# Папка для бинарника
MEGA_DIR="./megacmd"

# URL последней версии Generic Linux x64
MEGA_URL="https://mega.nz/linux/MEGAsync/Generic/megacmd-Linux-x64.tar.gz"

echo "Скачиваем MEGA-CMD..."
mkdir -p "$MEGA_DIR"
wget -O /tmp/megacmd.tar.gz "$MEGA_URL"

echo "Распаковываем..."
tar -xzf /tmp/megacmd.tar.gz -C "$MEGA_DIR" --strip-components=1

echo "Делаем бинарник исполняемым..."
chmod +x "$MEGA_DIR/mega"

echo "Готово! MEGA-CMD установлен в $MEGA_DIR"