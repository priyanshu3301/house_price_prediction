#!/usr/bin/env bash
#
# Deepnote-Optimized Ghidra Headless Analysis Script
#

set -e

# ==========================================
# CONFIGURATION
# ==========================================
# REPLACE THIS WITH YOUR DIRECT DOWNLOAD LINK FOR THE .SO FILE
TARGET_SO_URL="https://ps-index-drive.sad282.workers.dev/1:/libil2cpp.so"
SO_FILENAME="libil2cpp.so"

# Persistent directory in Deepnote workspace
WORK_DIR="/workspaces/ghidra_workspace"
PROJECT_NAME="HeadlessAnalysis"

# Resource configuration for 64GB RAM / 16 Cores
ALLOCATED_MAX_MEM="49152M"
CPU_CORES="16"

echo "================================================================"
echo "[+] Starting Ghidra Environment Setup on Deepnote Container"
echo "[*] Memory Target: ${ALLOCATED_MAX_MEM} | CPU Cores: ${CPU_CORES}"
echo "================================================================"

# ==========================================
# 1. UPDATE PACKAGES & INSTALL DEPENDENCIES
# ==========================================
echo "[+] Updating system package indexes and installing tools..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq openjdk-21-jdk wget unzip curl rsync zip jq

# Verify Java installation
echo "[+] Java environment verification:"
java -version

# ==========================================
# 2. FETCH LATEST GHIDRA AUTOMATICALLY
# ==========================================
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if [ ! -d "ghidra_latest" ]; then
    echo "[+] Dynamically resolving latest official Ghidra release URL..."
    
    # Query GitHub API for the latest public release download asset
    GHIDRA_DL_URL=$(curl -s https://api.github.com/repos/NationalSecurityAgency/ghidra/releases/latest \
        | jq -r '.assets[] | select(.name | test("ghidra_.*_PUBLIC_.*\\.zip$")) | .browser_download_url')

    if [ -z "${GHIDRA_DL_URL}" ] || [ "${GHIDRA_DL_URL}" = "null" ]; then
        echo "[-] Failed to automatically locate Ghidra release package via API."
        echo "[-] Using static fallback release mirror..."
        GHIDRA_DL_URL="https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.2_build/ghidra_11.2_PUBLIC_20240926.zip"
    fi

    echo "[+] Downloading Ghidra from: ${GHIDRA_DL_URL}"
    wget -q --show-progress "${GHIDRA_DL_URL}" -O ghidra.zip

    echo "[+] Extracting archive..."
    unzip -q ghidra.zip
    rm ghidra.zip

    # Rename extracted directory to a unified standard path
    EXTRACTED_DIR=$(ls -d ghidra_*_PUBLIC* | head -n 1)
    mv "${EXTRACTED_DIR}" ghidra_latest
fi

GHIDRA_DIR="${WORK_DIR}/ghidra_latest"

# Tune memory limits in launch properties
LAUNCH_PROPERTIES="${GHIDRA_DIR}/support/launch.properties"
if [ -f "${LAUNCH_PROPERTIES}" ]; then
    echo "[+] Configuring Java Max Heap to ${ALLOCATED_MAX_MEM} in launch settings..."
    sed -i "s/VMARG_MAXMEM=.*/VMARG_MAXMEM=${ALLOCATED_MAX_MEM}/" "${LAUNCH_PROPERTIES}"
fi

# ==========================================
# 3. DOWNLOAD TARGET BINARY
# ==========================================
mkdir -p "${WORK_DIR}/target"
cd "${WORK_DIR}/target"

echo "[+] Downloading shared object file..."
wget -q --show-progress "${TARGET_SO_URL}" -O "${SO_FILENAME}"

if [ ! -s "${SO_FILENAME}" ]; then
    echo "[-] Error: Downloaded file is empty or missing. Check TARGET_SO_URL."
    exit 1
fi
echo "[+] Binary acquired: $(du -h "${SO_FILENAME}" | cut -f1)"

# ==========================================
# 4. EXECUTE HEADLESS ANALYSIS
# ==========================================
PROJECT_DIR="${WORK_DIR}/projects"
mkdir -p "${PROJECT_DIR}"

echo "[+] Launching Ghidra Headless Analyzer..."
"${GHIDRA_DIR}/support/analyzeHeadless" \
    "${PROJECT_DIR}" \
    "${PROJECT_NAME}" \
    -import "${WORK_DIR}/target/${SO_FILENAME}" \
    -analysisTimeoutPerCpu 7200 \
    -max-cpu "${CPU_CORES}"

# ==========================================
# 5. PACKAGE PROCESSED PROJECT
# ==========================================
echo "[+] Packaging analysis results..."
cd "${PROJECT_DIR}"
zip -r -q "${WORK_DIR}/${PROJECT_NAME}_analyzed.zip" "${PROJECT_NAME}.gpr" "${PROJECT_NAME}.rep"

echo "================================================================"
echo "[SUCCESS] Analysis complete!"
echo "Processed Archive Path: ${WORK_DIR}/${PROJECT_NAME}_analyzed.zip"
echo "You can download this zip file directly from the Deepnote left sidebar."
echo "================================================================"
