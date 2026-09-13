from __future__ import annotations

import subprocess
import sys
import threading


def stream_reader(stream, log_file, output_stream):
    with open(log_file, "w", encoding="utf-8") as handle:
        for line in iter(stream.readline, ""):
            output_stream.write(line)
            handle.write(line)
    stream.close()


def run_logged_command(command, cwd, stdout_log, stderr_log):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        bufsize=1,
    )
    stdout_thread = threading.Thread(
        target=stream_reader, args=(process.stdout, stdout_log, sys.stdout), daemon=True
    )
    stderr_thread = threading.Thread(
        target=stream_reader, args=(process.stderr, stderr_log, sys.stderr), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()
    return process.wait()
