import cupy as cp

class KernelManager(object):
    """ Class for managing cupy RawKernels and allocated memory 
    
    Base class provides only:
        info() - print information about memory usage and CUDA block/grid size
    
    The subclass must provide:
        init() - (re)initialize kernel by allocating workspace memory and determining block/grid size
        execute() - function to execute the kernel
    
    """
    def __init__(self, name: str):
        self.workspace = {}
        self.name = name
        self._grid = None
        self._block = None
    
    def __repr__(self):
        return f"<KernelManager: {self.name}>"

    def info(self):
        """ Print information about memory usage and block size """
        mem_usage = 0
        print(f"--- {self.name} workspace ---")
        for k, v in self.workspace.items():
            print(f"{k}: \t {v.nbytes // 2**20} MiB")
            mem_usage += v.nbytes
        print(f"Total: \t {mem_usage // 2**20} MiB \n")
        
        print(f"--- {self.name} grid dimensions ---")
        print(f"Grid:  \t {self._grid}")
        print(f"Block: \t {self._block}\n")

    def init(self):
        """ This function should be supplied by the user, and should set:
        
        self._grid     - grid dims
        self._block    - block dims
        self.workspace - all cupy arrays required for workspace
        """
        raise NotImplementedError

    def execute(self):
        """ This function should be supplied by the user """
        raise NotImplementedError

    def __del__(self):
        """ Free memory when deleted 
        
        See https://docs.cupy.dev/en/stable/user_guide/memory.html
        """
        mempool = cp.get_default_memory_pool()
        for k, v in self.workspace.items():
            self.workspace[k] = None
        mempool.free_all_blocks()
        pass