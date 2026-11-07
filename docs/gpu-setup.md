# GPU machine setup (Windows + RTX 4070 12 GB → WSL2 → SSH from the Mac)

The route: **Mac → Windows OpenSSH (`gpu-box`) → WSL's own sshd on 127.0.0.1:2222 (`gpu-wsl`)**.
`gpu-box` is interactive-only on this machine; `gpu-wsl` supports one-off commands, rsync and tunnels.
Nothing listens on the LAN except the Windows sshd on port 22.

## 1. Windows

1. Update the NVIDIA driver. The Windows driver provides CUDA inside WSL; **do not** install a Linux NVIDIA driver in WSL.
2. `wsl --install -d Ubuntu-24.04`, then create the Linux user.
3. `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   memory=24GB
   swap=16GB
   localhostForwarding=true
   ```
4. Windows OpenSSH server with the Mac's key in `C:\ProgramData\ssh\administrators_authorized_keys` (admin accounts).
   Its `DefaultShell` is `C:\Windows\System32\wsl.exe`. **Limitation found on 2026-11-07:** `wsl.exe` rejects `-c`,
   so `ssh gpu-box 'cmd'` fails ("Invalid command line argument: -c") and rsync fails ("unexpected tag").
   `bash.exe` (the usual workaround) is not installed with current WSL. Hence section 3.
5. Power: Sleep and Hibernate on AC = Never. Pause Windows Update on training days.

## 2. Ubuntu (WSL2) basics

```bash
nvidia-smi                        # in an interactive shell; SSH sessions need the full path below
sudo apt update && sudo apt install -y rsync build-essential git tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
```
`/etc/wsl.conf` has `[boot] systemd=true`.

## 3. sshd inside WSL (127.0.0.1:2222, keys only)

```bash
. /etc/os-release && echo "$VERSION_ID"      # 24.04: socket settings are generated from sshd_config
sudo apt install -y openssh-server
sudo tee /etc/ssh/sshd_config.d/10-finsight.conf >/dev/null <<'EOF'
Port 2222
ListenAddress 127.0.0.1
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
AllowUsers harsha13
EOF
install -d -m 700 ~/.ssh && echo '<mac public key>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
sudo install -d -m 755 /run/sshd && sudo sshd -t && echo CONFIG_OK
sudo systemctl daemon-reload && sudo systemctl enable ssh.socket && sudo systemctl restart ssh.socket
systemctl cat ssh.socket | grep ListenStream   # only 127.0.0.1:2222
sudo ss -ltnp                                  # no 0.0.0.0 / [::] SSH listener
sudo sshd -T | grep -Ei '^(port|listenaddress|passwordauthentication|kbdinteractiveauthentication|permitrootlogin|allowusers) '
```

## 4. Mac `~/.ssh/config`

```
Host gpu-box
    HostName 192.168.0.128
    User ADMIN
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30

Host gpu-wsl
    HostName 127.0.0.1
    Port 2222
    User harsha13
    ProxyJump gpu-box
    HostKeyAlias gpu-wsl
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
```
Verified 2026-11-07: one-off commands, password login refused, rsync round trip with matching sha256
(binary file, space in a folder name, unicode, symlink), and `ssh -L 8000:127.0.0.1:8000 gpu-wsl` tunnelling.

SSH sessions into WSL do not get `/usr/lib/wsl/lib` (nvidia-smi) or `~/.local/bin` (uv) on `PATH`; CUDA libraries
still load through `ldconfig`. Remote commands therefore start with
`export PATH=/usr/lib/wsl/lib:$HOME/.local/bin:$PATH`.

## 5. Keeping WSL running (required for unattended training)

**Measured 2026-11-07:** WSL powered itself off (`systemd-poweroff` at 05:43:17 UTC) less than a minute after the
last Ubuntu window closed, killing every tmux session. tmux, systemd services and `gpu-wsl` SSH sessions do **not**
keep WSL alive; only a running `wsl.exe` process does. While WSL is stopped, `gpu-wsl` cannot connect
(`kex_exchange_identification: Connection closed`); any `ssh gpu-box` session starts it again.

Options, from simplest:
1. **Leave one Ubuntu window open** (minimised) for the whole run. Closing it stops training within a minute.
2. **A logon task that holds a hidden `wsl.exe`** (PowerShell as Administrator, one line at a time):
   ```powershell
   $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -Command `"Start-Process -WindowStyle Hidden -FilePath C:\Windows\System32\wsl.exe -ArgumentList '-d','Ubuntu-24.04','--exec','/bin/sleep','infinity'`""
   $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
   $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
   Register-ScheduledTask -TaskName "WSL keep-alive" -Action $action -Trigger $trigger -Settings $settings -Description "Keeps Ubuntu-24.04 (WSL) running for unattended FinSight training"
   Start-ScheduledTask -TaskName "WSL keep-alive"
   ```
   The task starts a hidden PowerShell that starts a hidden `wsl.exe` running `sleep infinity`, then exits
   (`LastTaskResult` 0); the keep-alive shows inside WSL as a `/bin/sleep infinity` process. This command line was
   checked on the PC on 2026-11-07 (with `touch` in place of `sleep`). A `conhost.exe --headless` launcher was also
   tried and started nothing from an SSH session, so it is not used.
   Remove with `Unregister-ScheduledTask -TaskName "WSL keep-alive" -Confirm:$false`, then end the keep-alive with
   `wsl.exe -d Ubuntu-24.04 --exec pkill -f "sleep infinity"` (WSL then stops within a minute if nothing else holds it).
   It needs the Windows user logged in; after a reboot without logon, WSL stays stopped and training resumes from the
   last checkpoint by hand.

Either way, verify before a long run: with every Ubuntu window and SSH session closed, the logger in
`~/finsight-logs/persist.log` (one line a minute, started with `tmux -L finsight`) must show an unchanged boot
time and no gaps for 30+ minutes. Checkpoints every 25 optimizer steps limit what any interruption costs.

The Linux clock in this WSL install jumps by about 25 seconds every few seconds (journal: "Clock change detected").
All durations in ModelForge use monotonic clocks; wall-clock timestamps are labels only.

## 6. What Claude runs (from the Mac)

```bash
R='export PATH=/usr/lib/wsl/lib:$HOME/.local/bin:$PATH; cd ~/developer/finsight'
git rev-parse HEAD > BUILD_COMMIT
rsync -az --delete --exclude .venv --exclude .git --exclude .legacy --exclude .aiagents --exclude .env \
  --exclude 'artifacts/adapters/' --exclude 'artifacts/probe/' --exclude 'artifacts/runs/' \
  --exclude 'artifacts/recovery/' --exclude 'artifacts/serving/' --exclude '*.log' ./ gpu-wsl:developer/finsight/
ssh gpu-wsl "$R && uv sync --locked --extra gpu && uv run --extra gpu pytest -q"

# Probe (stop vLLM first): real optimizer steps at the intended length, eval timing, adapter health, energy
ssh gpu-wsl "$R && uv run --extra gpu modelforge probe"

# Serve on 127.0.0.1 only, in tmux; tunnel from the Mac
ssh gpu-wsl "$R && tmux -L finsight new-session -d -s vllm 'uv run --extra gpu modelforge serve --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c 2>&1 | tee artifacts/serving/vllm.log'"
ssh -N -L 8000:127.0.0.1:8000 gpu-wsl &
uv run modelforge run --adapter qwen --model Qwen/Qwen3-4B --label qwen3-4b-base \
  --manifest finqa-grouped-comparison --selection val_shared_250

# Recovery test, then training in tmux (approval needed for the full run)
ssh gpu-wsl "$R && uv run --extra gpu python -m modelforge.training.recovery_test --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c"
ssh gpu-wsl "$R && tmux -L finsight new-session -d -s train 'uv run --extra gpu python -m modelforge.training.train_qlora --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --data artifacts/training/finqa-grouped-v1 --out artifacts/adapters/qwen3-4b-finqa-v1 --resume 2>&1 | tee -a artifacts/train.log'"
```

GPU energy is measured with NVML and is **GPU-only**. If the counter is unsupported under WSL, it is reported as
unavailable, never as zero; whole-system electricity needs a plug-in meter or a disclosed estimate.
