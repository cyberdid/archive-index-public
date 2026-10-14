"""Terminal check (plan F): run archive-tui in a real pseudo-terminal, type a query, render the screen with pyte.

usage: python tools/pty_tui_check.py [query] [out.txt]   (needs pyte; prints non-text symbols found)
"""
import os, pty, select, sys, time, struct, fcntl, termios
import pyte

COLS, ROWS = 140, 40
QUERY = sys.argv[1] if len(sys.argv) > 1 else "ретранслятор"
OUT = sys.argv[2] if len(sys.argv) > 2 else None

pid, fd = pty.fork()
if pid == 0:
    os.environ["TERM"] = "xterm-256color"
    os.environ["PYTHONPATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    os.execv(sys.executable, [sys.executable, "-m", "archive_index.tui"])
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
screen = pyte.Screen(COLS, ROWS)
stream = pyte.ByteStream(screen)

def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try:
                data = os.read(fd, 65536)
            except OSError:
                return False
            if not data:
                return False
            stream.feed(data)
    return True

pump(25)  # startup: model load and first paint
os.write(fd, b"/")
pump(1)
os.write(fd, QUERY.encode("utf-8"))
pump(1)
os.write(fd, b"\r")
EXTRA = sys.argv[3] if len(sys.argv) > 3 else ""
pump(60 if EXTRA else 25)  # a first search loads the model and may take half a minute
EXTRA = sys.argv[3] if len(sys.argv) > 3 else ""
if EXTRA:  # e.g. "g" to build the neuron picture: Esc first moves focus from the query field to the results
    os.write(fd, b"\x1b")
    pump(1)
    os.write(fd, EXTRA.encode("utf-8"))
    pump(20)
lines = [screen.display[i].rstrip() for i in range(ROWS)]
text = "\n".join(lines)
print(text)
if OUT:
    open(OUT, "w", encoding="utf-8").write(text)
odd = sorted({c for c in text if ord(c) > 127 and c not in "абвгґдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"})
print("\nNON-CYRILLIC NON-ASCII CHARS:", [f"{c} U+{ord(c):04X}" for c in odd])
try:
    os.kill(pid, 9)
except OSError:
    pass
