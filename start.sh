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
# 1. INSTALL BASE SYSTEM UTILITIES
# ==========================================
echo "[+] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget unzip curl rsync zip jq

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# ==========================================
# 2. DOWNLOAD PORTABLE OPENJDK 21
# ==========================================
JDK_DIR="${WORK_DIR}/jdk-21"
if [ ! -d "${JDK_DIR}" ] || [ ! -f "${JDK_DIR}/bin/java" ]; then
    echo "[+] Downloading official OpenJDK 21 binary..."
    
    JDK_URL="https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse"
    
    rm -rf "${JDK_DIR}" openjdk21.tar.gz
    curl -L --progress-bar "${JDK_URL}" -o openjdk21.tar.gz
    
    echo "[+] Extracting OpenJDK 21..."
    mkdir -p "${JDK_DIR}"
    tar -xzf openjdk21.tar.gz -C "${JDK_DIR}" --strip-components=1
    rm openjdk21.tar.gz
fi

export JAVA_HOME="${JDK_DIR}"
export PATH="${JAVA_HOME}/bin:${PATH}"

echo "[+] Verified Java installation at ${JAVA_HOME}:"
"${JAVA_HOME}/bin/java" -version

# ==========================================
# 3. DOWNLOAD & CONFIGURE GHIDRA
# ==========================================
if [ ! -d "ghidra_latest" ]; then
    echo "[+] Resolving latest Ghidra release archive..."
    
    GHIDRA_DL_URL=$(curl -s https://api.github.com/repos/NationalSecurityAgency/ghidra/releases/latest \
        | jq -r '.assets[] | select(.name | test("ghidra_.*_PUBLIC_.*\\.zip$")) | .browser_download_url')

    if [ -z "${GHIDRA_DL_URL}" ] || [ "${GHIDRA_DL_URL}" = "null" ]; then
        echo "[-] Using fallback release URL..."
        GHIDRA_DL_URL="https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.2_build/ghidra_11.2_PUBLIC_20240926.zip"
    fi

    echo "[+] Downloading Ghidra from: ${GHIDRA_DL_URL}"
    wget -q --show-progress "${GHIDRA_DL_URL}" -O ghidra.zip

    echo "[+] Extracting Ghidra..."
    unzip -q ghidra.zip
    rm ghidra.zip

    EXTRACTED_DIR=$(ls -d ghidra_*_PUBLIC* | head -n 1)
    mv "${EXTRACTED_DIR}" ghidra_latest
fi

GHIDRA_DIR="${WORK_DIR}/ghidra_latest"

LAUNCH_PROPERTIES="${GHIDRA_DIR}/support/launch.properties"
if [ -f "${LAUNCH_PROPERTIES}" ]; then
    echo "[+] Configuring Java Max Heap to ${ALLOCATED_MAX_MEM} in launch settings..."
    sed -i "s/VMARG_MAXMEM=.*/VMARG_MAXMEM=${ALLOCATED_MAX_MEM}/" "${LAUNCH_PROPERTIES}"
fi

# ==========================================
# 4. FETCH TARGET BINARY WITH STRICT CHECKING
# ==========================================
TARGET_DIR="${WORK_DIR}/target"
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

FULL_SO_PATH="${TARGET_DIR}/${SO_FILENAME}"

if [ ! -f "${FULL_SO_PATH}" ] || [ ! -s "${FULL_SO_PATH}" ]; then
    echo "[+] Downloading target shared object binary..."
    curl -L --progress-bar -o "${FULL_SO_PATH}" "${TARGET_SO_URL}"
fi

if [ ! -s "${FULL_SO_PATH}" ]; then
    echo "[-] FATAL ERROR: Binary file does not exist or is empty at ${FULL_SO_PATH}"
    echo "[-] Please check your TARGET_SO_URL inside start.sh"
    exit 1
fi

echo "[+] Binary acquired: $(du -h "${FULL_SO_PATH}" | cut -f1)"

# ==========================================
# 5. EXECUTE HEADLESS ANALYSIS
# ==========================================
PROJECT_DIR="${WORK_DIR}/projects"
mkdir -p "${PROJECT_DIR}"

echo "[+] Launching Ghidra Headless Analyzer..."

JAVA_HOME="${JDK_DIR}" PATH="${JDK_DIR}/bin:${PATH}" \
"${GHIDRA_DIR}/support/analyzeHeadless" \
    "${PROJECT_DIR}" \
    "${PROJECT_NAME}" \
    -import "${FULL_SO_PATH}" \
    -max-cpu "${CPU_CORES}"

# ==========================================
# 6. PACKAGE PROCESSED PROJECT
# ==========================================
echo "[+] Packaging analysis results..."
cd "${PROJECT_DIR}"
zip -r -q "${WORK_DIR}/${PROJECT_NAME}_analyzed.zip" "${PROJECT_NAME}.gpr" "${PROJECT_NAME}.rep"

echo "================================================================"
echo "[SUCCESS] Analysis complete!"
echo "Processed Archive Path: ${WORK_DIR}/${PROJECT_NAME}_analyzed.zip"
echo "You can download this zip file directly from the Deepnote left sidebar."
echo "================================================================"
