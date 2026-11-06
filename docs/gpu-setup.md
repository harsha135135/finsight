# GPU machine setup (Windows + RTX 4070 12 GB → WSL2 → SSH from the Mac)

About 30–45 minutes the first time. The aim: from the Mac, `ssh gpu-box` lands in a WSL2 Ubuntu shell where `nvidia-smi` works.

## 1. Windows (PowerShell **as Administrator**)

1. Update the NVIDIA driver (GeForce Experience or nvidia.com). The Windows driver provides CUDA inside WSL. **Do not** install a Linux NVIDIA driver inside WSL.
2. Install WSL2 with Ubuntu, then reboot and create your Linux user when prompted:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
3. Give WSL enough memory. Create `C:\Users\<you>\.wslconfig` (adjust to your RAM; leave Windows at least 8 GB):
   ```ini
   [wsl2]
   memory=24GB
   swap=16GB
   ```
   Then run `wsl --shutdown` and reopen Ubuntu.
4. Install the Windows OpenSSH server and make SSH sessions open the Ubuntu shell:
   ```powershell
   Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
   Start-Service sshd
   Set-Service -Name sshd -StartupType Automatic
   New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\bash.exe" -PropertyType String -Force
   ```
   `bash.exe` starts your default WSL distro and accepts `-c`, which `rsync` and `ssh host 'command'` need.
5. Add the Mac's public key. Copy the output of `cat ~/.ssh/id_ed25519.pub` on the Mac; create a key first with `ssh-keygen -t ed25519` if you don't have one. Then:
   - **If your Windows account is an administrator** (the usual case), the key must go in the administrators file, with restricted permissions:
     ```powershell
     Add-Content -Path C:\ProgramData\ssh\administrators_authorized_keys -Value "ssh-ed25519 AAAA... you@mac"
     icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
     ```
   - **If it is a standard account:** put the key in `C:\Users\<you>\.ssh\authorized_keys` instead.
6. Note the PC's LAN IP (`ipconfig`, the IPv4 address of your Wi-Fi or Ethernet adapter). In Power settings, set **Sleep: Never** while plugged in, so training isn't interrupted.

## 2. Inside Ubuntu (WSL2)

```bash
nvidia-smi                        # must show the RTX 4070
sudo apt update && sudo apt install -y rsync build-essential git
curl -LsSf https://astral.sh/uv/install.sh | sh
df -h ~                           # want 60+ GB free (model downloads + checkpoints)
```

## 3. On the Mac

Add this to `~/.ssh/config`:
```
Host gpu-box
    HostName 192.168.x.y        # the PC's LAN IP
    User <windows-username>
    ServerAliveInterval 30
```
Test: `ssh gpu-box 'uname -a && nvidia-smi --query-gpu=name,memory.total --format=csv'` should print a Linux kernel line and `NVIDIA GeForce RTX 4070, 12282 MiB`.

**Then tell Claude the host alias is ready.** Everything below runs from the Mac.

## 4. What Claude will run (for reference)

```bash
# Sync code and frozen training data (no secrets, no virtualenv)
rsync -az --delete --exclude .venv --exclude .git --exclude .legacy --exclude .aiagents ./ gpu-box:~/finsight/
ssh gpu-box 'cd ~/finsight && ~/.local/bin/uv sync --extra gpu'

# 1) Probe: VRAM, telemetry, 4-bit loading, QLoRA step memory by sequence length
ssh gpu-box 'cd ~/finsight && ~/.local/bin/uv run --extra gpu modelforge probe'

# 2) Serve the unchanged baseline with vLLM, reached over an SSH tunnel (no open port)
ssh gpu-box 'cd ~/finsight && nohup ~/.local/bin/uv run --extra gpu vllm serve Qwen/Qwen3.5-4B --revision <sha> \
   --max-model-len 4096 --enable-lora --max-lora-rank 16 > vllm.log 2>&1 &'
ssh -N -L 8000:localhost:8000 gpu-box &
uv run modelforge run --adapter qwen --model Qwen/Qwen3.5-4B --label qwen35-4b-baseline --selection val_200

# 3) QLoRA fine-tune on the frozen train split, then serve the adapter on the same base
ssh gpu-box 'cd ~/finsight && ~/.local/bin/uv run --extra gpu python -m modelforge.training.train_qlora \
   --model Qwen/Qwen3.5-4B --revision <sha> --data artifacts/training/finqa-grouped-v1 --out artifacts/adapters/qwen35-4b-finqa-v1'
```

Notes:
- If the vLLM server can't be reached through the tunnel, check that WSL's localhost forwarding is on (the default) or use `networkingMode=mirrored` in `.wslconfig` (Windows 11 22H2+).
- GPU energy is measured with NVML and is **GPU-only**. The probe reports whether the energy counter works under your WSL install. If it doesn't, the value is reported as unavailable, never as zero. For whole-system electricity, a plug-in power meter is the only true measurement; otherwise we use a disclosed estimate.
