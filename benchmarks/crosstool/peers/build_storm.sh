#!/usr/bin/env bash
# Build Storm from source inside the impact_work container (Debian 12, gcc-12).
# LTO is disabled to avoid the high-core link-time OOM the recon flagged; we build
# only the `storm` CLI target. Logs to /var/log/storm_build.log.
set -e
LOG=/var/log/storm_build.log
exec > >(tee -a "$LOG") 2>&1
echo "=== storm build start: $(date) ==="

echo "--- apt deps ---"
apt-get update -qq
apt-get install -y --no-install-recommends \
    automake build-essential cmake git \
    libboost-all-dev libcln-dev libginac-dev libgmp-dev libglpk-dev \
    libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev libarchive-dev

echo "--- clone storm (stable) ---"
if [ ! -d /opt/storm/.git ]; then
    git clone --depth 1 -b stable https://github.com/moves-rwth/storm.git /opt/storm
fi

echo "--- cmake configure (Release, LTO off) ---"
mkdir -p /opt/storm/build
cd /opt/storm/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSTORM_USE_LTO=OFF

echo "--- build storm CLI (-j16 to bound peak RAM) ---"
make storm-cli -j16

echo "--- link into PATH ---"
ln -sf /opt/storm/build/bin/storm /usr/local/bin/storm
/usr/local/bin/storm --version
echo "=== storm build done: $(date) ==="
