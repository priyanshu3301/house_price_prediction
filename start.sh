#!/usr/bin/env bash
#
# Ghidra Headless Analysis Script - 64GB RAM / 16 Core Setup
#

set -e

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_SO_URL="https://ps-index-drive.sad282.workers.dev/1:/libil2cpp.so"
SO_FILENAME="libtarget.so"

GHIDRA_VER="11.2"
GHIDRA_DATE="20240926"
GHIDRA_URL="https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.2_build/ghidra_11.2_PUBLIC_${GHIDRA_DATE}.zip"

WORK_DIR="/tmp/ghidra_workspace"
PROJECT_NAME="HeadlessAnalysis"

# ==========================================
# 1. HARDWARE ALLOCATION
# ==========================================
# Allocating ~48GB (75%) for Java Heap, keeping ~16GB free for system overhead
ALLOCATED_MAX_MEM="49152M"
CPU_CORES="16"

echo "[*] Configured Java Max Heap: ${ALLOCATED_MAX_MEM}"
echo "[*] Configured Threads: ${CPU_CORES}"

echo "[+] Updating system packages and installing Java..."
apt-get update -qq && apt-get install -y -qq openjdk-21-jdk wget unzip curl rsync zip

# ==========================================
# 2. CONFIGURE GHIDRA
# ==========================================
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if [ ! -d "ghidra_${GHIDRA_VER}_PUBLIC" ]; then
    echo "[+] Downloading Ghidra v${GHIDRA_VER}..."
    wget -q "${GHIDRA_URL}" -O ghidra.zip
    unzip -q ghidra.zip
    rm ghidra.zip
fi

GHIDRA_DIR="${WORK_DIR}/ghidra_${GHIDRA_VER}_PUBLIC"

# Update launch.properties with allocated heap memory
sed -i "s/VMARG_MAXMEM=.*/VMARG_MAXMEM=${ALLOCATED_MAX_MEM}/" "${GHIDRA_DIR}/support/launch.properties"

# ==========================================
# 3. DOWNLOAD TARGET & RUN HEADLESS ANALYSIS
# ==========================================
mkdir -p "${WORK_DIR}/target"
cd "${WORK_DIR}/target"
echo "[+] Fetching binary..."
wget -q "${TARGET_SO_URL}" -O "${SO_FILENAME}"

PROJECT_DIR="${WORK_DIR}/projects"
mkdir -p "${PROJECT_DIR}"

echo "[+] Starting Ghidra auto-analysis on ${CPU_CORES} threads..."
"${GHIDRA_DIR}/support/analyzeHeadless" \
    "${PROJECT_DIR}" \
    "${PROJECT_NAME}" \
    -import "${WORK_DIR}/target/${SO_FILENAME}" \
    -analysisTimeoutPerCpu 7200 \
    -max-cpu "${CPU_CORES}"

# ==========================================
# 4. PACKAGE FOR DOWNLOAD
# ==========================================
echo "[+] Packaging project files into ZIP archive..."
cd "${PROJECT_DIR}"
zip -r -q "${WORK_DIR}/${PROJECT_NAME}_analyzed.zip" "${PROJECT_NAME}.gpr" "${PROJECT_NAME}.rep"

echo "================================================================"
echo "[SUCCESS] Analysis complete!"
echo "Download path: ${WORK_DIR}/${PROJECT_NAME}_analyzed.zip"
echo "================================================================"
