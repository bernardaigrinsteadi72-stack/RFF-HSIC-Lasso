"""Windows process memory guard. The handle must stay alive for the entire run."""
import ctypes as C
from ctypes import wintypes as W
import os

_job=None

class Basic(C.Structure):
    _fields_=[('PerProcessUserTimeLimit',C.c_int64),('PerJobUserTimeLimit',C.c_int64),
              ('LimitFlags',W.DWORD),('MinimumWorkingSetSize',C.c_size_t),
              ('MaximumWorkingSetSize',C.c_size_t),('ActiveProcessLimit',W.DWORD),
              ('Affinity',C.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
class IO(C.Structure):
    _fields_=[(s,C.c_uint64) for s in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
class Extended(C.Structure):
    _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',C.c_size_t),
              ('JobMemoryLimit',C.c_size_t),('PeakProcessMemoryUsed',C.c_size_t),('PeakJobMemoryUsed',C.c_size_t)]
class Memory(C.Structure):
    _fields_=[('cb',W.DWORD),('PageFaultCount',W.DWORD)]+[(s,C.c_size_t) for s in
              ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]

def memory_info():
    if os.name!='nt': return {}
    k=C.WinDLL('kernel32',use_last_error=True); p=C.WinDLL('psapi',use_last_error=True)
    k.GetCurrentProcess.restype=W.HANDLE
    p.GetProcessMemoryInfo.argtypes=[W.HANDLE,C.POINTER(Memory),W.DWORD]
    m=Memory(); m.cb=C.sizeof(m)
    if not p.GetProcessMemoryInfo(k.GetCurrentProcess(),C.byref(m),m.cb): raise C.WinError(C.get_last_error())
    return {'working_set_mib':m.WorkingSetSize/2**20,'peak_working_set_mib':m.PeakWorkingSetSize/2**20,
            'private_mib':m.PrivateUsage/2**20,'peak_commit_mib':m.PeakPagefileUsage/2**20}

def enforce_memory_limit(mib=4096):
    global _job
    if os.name!='nt': raise RuntimeError('This resource guard targets Windows')
    k=C.WinDLL('kernel32',use_last_error=True)
    k.CreateJobObjectW.argtypes=[C.c_void_p,W.LPCWSTR]; k.CreateJobObjectW.restype=W.HANDLE
    k.GetCurrentProcess.restype=W.HANDLE
    k.SetInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD]
    k.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE]
    _job=k.CreateJobObjectW(None,None)
    e=Extended(); e.BasicLimitInformation.LimitFlags=0x100 # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    e.ProcessMemoryLimit=int(mib*2**20)
    if not _job or not k.SetInformationJobObject(_job,9,C.byref(e),C.sizeof(e)):
        raise C.WinError(C.get_last_error())
    if not k.AssignProcessToJobObject(_job,k.GetCurrentProcess()): raise C.WinError(C.get_last_error())
    return {'enforced_process_commit_limit_mib':mib}

