"""The verifier for reinforcement learning: run generated code against hidden tests in a sandbox.

``check(solution, test)`` returns True when every test passes. Each check runs in
a fresh interpreter in an empty temporary directory, isolated as far as plain
Python allows:

- ``python -I -B -S``: no environment variables, user site, script directory or
  bytecode writes; an empty environment apart from ``SYSTEMROOT``;
- an audit hook installed before the generated code is imported refuses network
  and process modules (``socket``, ``subprocess``, ``ctypes``, ``multiprocessing``,
  ``urllib``, ...), the events behind them (``socket.*``, ``os.system``,
  ``os.exec*``, ``os.spawn*``, ``os.kill``, ...), every file write, delete,
  rename or directory change, and ``ctypes``;
- resource limits: on Windows a job object (memory per process, CPU time, a
  single active process so nothing can be spawned, every process killed when the
  check ends), elsewhere ``setrlimit`` (address space, CPU time, no new processes,
  no file writes); a wall-clock timeout on top;
- the runner learns a random nonce on stdin before any generated code runs and
  prints it only after every test passed, so output from the code itself
  cannot fake a pass, and neither can ``exit(0)``.

Tests are either a module of top-level asserts (MBPP) or pytest-style
``test_*`` functions with ``pytest.raises``, ``approx`` and
``mark.parametrize`` (KodCode; a shim stands in for pytest). The candidate
module is ``solution``.
"""
import concurrent.futures
import os
import secrets
import subprocess
import sys
import tempfile

from .sft_data import PYTEST_SHIM

MEMORY_MB = 512
CPU_SECONDS = 10.0   # a backstop: Windows enforces job time limits only every few seconds
TIMEOUT = 5.0
# a venv's python.exe on Windows is a launcher that starts the base interpreter as a
# child process, which the one-process limit forbids
PYTHON = getattr(sys, '_base_executable', None) or sys.executable

RUNNER = r'''import sys
_nonce = sys.stdin.readline().strip()
sys.path.insert(0, sys.argv[1])
import inspect


def _install():
    import os
    bad_modules = frozenset((
        'socket', '_socket', 'ssl', '_ssl', 'select', 'selectors', 'subprocess',
        '_posixsubprocess', '_winapi', 'multiprocessing', '_multiprocessing', 'ctypes',
        '_ctypes', 'urllib', 'http', 'ftplib', 'smtplib', 'telnetlib', 'xmlrpc', 'asyncio',
        '_overlapped', '_asyncio', 'winreg', 'msvcrt', 'mmap', 'webbrowser', 'concurrent',
        'pty', 'resource', 'sqlite3', '_sqlite3'))
    bad_events = (
        'socket.', 'subprocess.', 'os.system', 'os.exec', 'os.spawn', 'os.posix_spawn',
        'os.startfile', 'os.kill', 'os.fork', 'os.forkpty', 'os.remove', 'os.unlink',
        'os.rename', 'os.replace', 'os.rmdir', 'os.removedirs', 'os.mkdir', 'os.makedirs',
        'os.chmod', 'os.chown', 'os.link', 'os.symlink', 'os.truncate', 'os.putenv',
        'os.unsetenv', 'os.chdir', 'os.chflags', 'os.utime', 'os.add_dll_directory',
        'os.setxattr', 'os.removexattr', 'shutil.', 'ctypes.', 'winreg.', 'urllib.',
        'webbrowser.', '_winapi.', 'msvcrt.', 'mmap.', 'sqlite3.', 'signal.', 'faulthandler.')
    write = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

    def hook(event, args):
        if event == 'import':
            if args[0].partition('.')[0] in bad_modules:
                raise ImportError('blocked in the sandbox: ' + args[0])
        elif event == 'open':
            mode, flags = args[1], args[2]
            if ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                    or (isinstance(flags, int) and flags & write)):
                raise PermissionError('blocked in the sandbox: writing files')
        elif event.startswith(bad_events):
            raise PermissionError('blocked in the sandbox: ' + event)

    sys.addaudithook(hook)


def _run(nonce):
    import test_solution as t     # top-level asserts (MBPP) run here
    n = 0
    for name, f in list(vars(t).items()):
        if not name.startswith('test') or not callable(f) or isinstance(f, type):
            continue
        params = getattr(f, '_params', None)
        if params:
            names, values = params
            for v in values:
                v = v if len(names) > 1 else (v,)
                f(**dict(zip(names, v)))
                n += 1
        else:
            if inspect.signature(f).parameters:
                raise SystemExit('fixture')
            f()
            n += 1
    out = sys.__stdout__
    out.write('\nPASS ' + nonce + ' ' + str(n) + '\n')
    out.flush()


_install()
del _install
try:
    _run(_nonce)
except BaseException:
    import traceback
    traceback.print_exc()
    sys.__stdout__.flush()
    import os
    os._exit(1)
'''


# -- resource limits ---------------------------------------------------------------

if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL('kernel32', use_last_error=True)

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
            'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64),
                    ('PerJobUserTimeLimit', ctypes.c_int64),
                    ('LimitFlags', wintypes.DWORD),
                    ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t),
                    ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t),
                    ('PriorityClass', wintypes.DWORD),
                    ('SchedulingClass', wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', _BasicLimits), ('IoInfo', _IoCounters),
                    ('ProcessMemoryLimit', ctypes.c_size_t),
                    ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t),
                    ('PeakJobMemoryUsed', ctypes.c_size_t)]

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                             wintypes.DWORD)
    _k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)

    def _limit(proc):
        """A job object for ``proc``: memory, CPU time, one process, killed on close."""
        job = _k32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _ExtendedLimits()
        b = info.BasicLimitInformation
        b.LimitFlags = 0x2 | 0x8 | 0x100 | 0x400 | 0x2000   # time, processes, memory,
        b.PerProcessUserTimeLimit = int(CPU_SECONDS * 1e7)   # no error boxes, kill on close
        b.ActiveProcessLimit = 1
        info.ProcessMemoryLimit = MEMORY_MB << 20
        if not (_k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
                and _k32.AssignProcessToJobObject(job, int(proc._handle))):
            err = ctypes.get_last_error()
            _k32.CloseHandle(job)
            proc.kill()
            raise ctypes.WinError(err)
        return job

    def _release(job):
        _k32.CloseHandle(job)   # kills whatever is left in the job

    _preexec = None
else:
    def _preexec():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_MB << 20,) * 2)
        cpu = int(CPU_SECONDS + 0.999)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

    def _limit(proc):
        return None

    def _release(job):
        pass


# -- checks ------------------------------------------------------------------------

def run(solution, test, timeout=TIMEOUT):
    """``(passed, number of test functions, stderr tail)`` of one check."""
    nonce = secrets.token_hex(16)
    with tempfile.TemporaryDirectory(prefix='vabs_verify_') as d:
        files = {'solution.py': solution, 'test_solution.py': test, 'pytest.py': PYTEST_SHIM,
                 'runner.py': RUNNER}
        for name, text in files.items():
            with open(os.path.join(d, name), 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
        out_path, err_path = os.path.join(d, 'out.txt'), os.path.join(d, 'err.txt')
        with open(out_path, 'wb') as out, open(err_path, 'wb') as err:
            proc = subprocess.Popen(
                [PYTHON, '-I', '-B', '-S', '-X', 'utf8', os.path.join(d, 'runner.py'), d],
                cwd=d, stdin=subprocess.PIPE, stdout=out, stderr=err, preexec_fn=_preexec,
                env={'SYSTEMROOT': os.environ.get('SYSTEMROOT', ''), 'PYTHONHASHSEED': '0'},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            job = None
            try:
                job = _limit(proc)
                try:
                    proc.communicate((nonce + '\n').encode(), timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            finally:
                if job is not None:
                    _release(job)
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        with open(out_path, 'rb') as f:
            f.seek(max(0, os.path.getsize(out_path) - 4096))
            tail = f.read().decode('utf-8', 'replace')
        with open(err_path, 'rb') as f:
            f.seek(max(0, os.path.getsize(err_path) - 2048))
            err_tail = f.read().decode('utf-8', 'replace')
    mark = f'PASS {nonce} '
    for line in tail.splitlines():
        if line.startswith(mark) and proc.returncode == 0:
            return True, int(line[len(mark):]), err_tail
    return False, 0, err_tail or ('timeout' if proc.returncode is None else '')


def check(solution, test, timeout=TIMEOUT):
    return run(solution, test, timeout)[0]


class Pool:
    """Run many checks at once (each in its own sandboxed process)."""

    def __init__(self, workers=None):
        self.ex = concurrent.futures.ThreadPoolExecutor(workers or min(32, os.cpu_count() or 4))

    def map(self, pairs, timeout=TIMEOUT):
        """``[passed]`` for ``(solution, test)`` pairs, in order."""
        return list(self.ex.map(lambda p: check(p[0], p[1], timeout), pairs))

    def close(self):
        self.ex.shutdown()
