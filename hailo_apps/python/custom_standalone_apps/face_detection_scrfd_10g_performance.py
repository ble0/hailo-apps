#!/usr/bin/env python3
# sudo pkill -f face_detection_scrfd_10g_performance.py

"""
Performance tracking script with separate process and socket health monitoring.

USAGE:
    python face_detection_scrfd_10g_performance.py --process "SUBSET_OF_COMMAND" [--port PORT] [--interval SECONDS]

EXAMPLE:
    python face_detection_scrfd_10g_performance.py --process "face_detection_scrfd_10g.py"

The script will:
- Find exactly ONE running process whose command line contains the given substring.
- If zero or multiple matches, it will list them and exit.
- Then monitor that process, hardware stats, socket on specified port, and kernel alerts.
"""

import csv
import os
import subprocess
import time
import argparse
from datetime import datetime
import psutil

# ==================== CONFIGURATION (via arguments) ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor a process and system stats on Raspberry Pi 5.")
    parser.add_argument("--process", "-p", required=True,
                        help="Substring to match in command line of target process (must match exactly one).")
    parser.add_argument("--port", "-P", type=int, default=8080,
                        help="TCP port to check for socket binding/connections (default: 8080).")
    parser.add_argument("--interval", "-i", type=int, default=300,
                        help="Logging interval in seconds (default: 300).")
    parser.add_argument("--log", "-l", default="pi5_performance_log.csv",
                        help="Output CSV log file path (default: pi5_performance_log.csv).")
    return parser.parse_args()

# ==================== HELPER FUNCTIONS ====================

def get_vcgen_data(command: str) -> str:
    try:
        result = subprocess.check_output(["vcgencmd", command]).decode("utf-8")
        return result.strip().split("=")[1]
    except (subprocess.CalledProcessError, IndexError, OSError):
        return "N/A"

def get_socket_status(port: int) -> str:
    """
    Check if a TCP port is in LISTEN state and if there are established connections.
    Returns: "LISTENING", "ESTABLISHED", or "DOWN"
    """
    listening = False
    established = False
    try:
        for conn in psutil.net_connections(kind='tcp'):
            if conn.laddr.port == port:
                if conn.status == 'LISTEN':
                    listening = True
                elif conn.status == 'ESTABLISHED':
                    established = True
    except (psutil.AccessDenied, OSError):
        return "ERROR"
    
    if established:
        return "ESTABLISHED"
    elif listening:
        return "LISTENING"
    else:
        return "DOWN"

def find_unique_process(target_substring):
    current_pid = os.getpid()
    matches = []
    for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
        try:
            if proc.info["pid"] == current_pid:
                continue   # skip self
            cmdline = proc.info["cmdline"]
            if cmdline and target_substring in " ".join(cmdline):
                matches.append((proc.info["pid"], " ".join(cmdline)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches

def get_process_stats(proc):
    if proc is None:
        return (0, 0)
    try:
        cpu = proc.cpu_percent(interval=0)
        mem = proc.memory_info().rss / (1024 * 1024)
        return (cpu, mem)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return (0, 0)

def get_kernel_alerts() -> str:
    try:
        raw = subprocess.check_output(["sudo", "dmesg", "-c"]).decode("utf-8").lower()
        if not raw:
            return "NONE"
        alerts = []
        if "usb" in raw and "disconnect" in raw:
            alerts.append("USB_DISCONNECT")
        if "under-voltage" in raw:
            alerts.append("LOW_VOLTAGE")
        if "oom-killer" in raw:
            alerts.append("OUT_OF_MEMORY")
        return "|".join(alerts) if alerts else "OTHER_KERN_EVENT"
    except (subprocess.CalledProcessError, OSError):
        return "ERR_DMESG"

# ==================== MAIN ====================

def main():
    args = parse_args()
    
    # Clear kernel logs at start
    subprocess.call(["sudo", "dmesg", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # --- Find target process ---
    matches = find_unique_process(args.process)
    if len(matches) == 0:
        print(f"ERROR: No running process found with command line containing '{args.process}'")
        print("Make sure your target process is running.")
        sys.exit(1)
    elif len(matches) > 1:
        print(f"ERROR: Multiple processes match '{args.process}':")
        for pid, cmdline in matches:
            print(f"  PID {pid}: {cmdline}")
        print("\nPlease refine your --process substring to match exactly one process.")
        sys.exit(1)
    
    # Exactly one match
    target_pid, target_cmdline = matches[0]
    print(f"Found target process: PID {target_pid} | {target_cmdline}")
    # Get process object for monitoring
    target_proc = psutil.Process(target_pid)
    
    # --- Prepare CSV headers ---
    core_count = psutil.cpu_count()
    headers = [
        "Timestamp", "Process_Status", "Socket_Status", "Clients",
        "Temp(C)", "Volt", "Freq(MHz)", "Throttled", "Kernel_Alert",
    ]
    per_core_headers = [f"Core{i}_%" for i in range(core_count)]
    headers.extend(per_core_headers)
    headers.extend(["Sys_CPU_%", "Sys_Mem_%", "Load_1m", "Load_5m", "Load_15m", "Proc_CPU_%", "Proc_Mem_MB"])
    
    print(f"Monitoring process: {target_cmdline}")
    print(f"Monitoring socket on port: {args.port}")
    print(f"Logging to {args.log} every {args.interval} seconds.")
    print("Press Ctrl+C to stop.\n")
    
    # --- Main loop ---
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Process status (re-check if still alive)
        try:
            target_proc = psutil.Process(target_pid)
            if target_proc.is_running():
                process_status = "ALIVE"
            else:
                process_status = "DEAD"
                target_proc = None
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_status = "DEAD"
            target_proc = None
        
        socket_status = get_socket_status(args.port)
        
        # Count established clients
        client_count = 0
        try:
            for conn in psutil.net_connections(kind='tcp'):
                if conn.laddr.port == args.port and conn.status == 'ESTABLISHED':
                    client_count += 1
        except:
            pass
        
        # --- Hardware ---
        temp = get_vcgen_data("measure_temp").replace("'C", "")
        volt = get_vcgen_data("measure_volts core").replace("V", "")
        freq_raw = get_vcgen_data("measure_clock arm")
        freq_mhz = int(freq_raw) // 1_000_000 if freq_raw != "N/A" else "N/A"
        throttled = get_vcgen_data("get_throttled")
        kernel_alerts = get_kernel_alerts()
        
        # --- System stats ---
        sys_cpu = psutil.cpu_percent(interval=0)
        sys_mem = psutil.virtual_memory().percent
        load_avg = os.getloadavg()
        per_cpu = psutil.cpu_percent(interval=0, percpu=True)
        
        # --- Process stats ---
        proc_cpu, proc_mem = get_process_stats(target_proc)
        
        # --- Build row ---
        row = [timestamp, process_status, socket_status, client_count,
               temp, volt, freq_mhz, throttled, kernel_alerts]
        row.extend(per_cpu)
        row.extend([sys_cpu, sys_mem, load_avg[0], load_avg[1], load_avg[2], proc_cpu, proc_mem])
        
        # --- Write CSV ---
        file_exists = os.path.isfile(args.log)
        with open(args.log, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
        
        # --- Console feedback ---
        core_parts = [f"C{i}:{val:>3}%" for i, val in enumerate(per_cpu)]
        core_str = " ".join(core_parts)
        print(f"[{timestamp}] {process_status:<5} | "
              f"Socket:{socket_status:<11} | "
              f"Clients:{client_count:>2} | "
              f"SysCPU:{sys_cpu:>4}% | "
              f"ProcCPU:{proc_cpu:>4}% | "
              f"Temp:{temp:>5}C | "
              f"{core_str}")
        
        time.sleep(args.interval)

if __name__ == "__main__":
    import sys
    main()