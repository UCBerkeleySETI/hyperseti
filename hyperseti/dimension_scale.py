from astropy.units import Unit, Quantity
from astropy.time import Time, TimeDelta
import numpy as np

HANDLED_FUNCTIONS = {}

def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects.
    
    See https://numpy.org/neps/nep-0018-array-function-protocol.html
    """
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

def isscalar(x):
    """ Check if x is scalar """
    # Quantity does not return True with np.isscalar, even if q.isscalar=True
    if isinstance(x, (Quantity, Time, TimeDelta)):
        return x.isscalar
    else:
        return np.isscalar(x)

def issamelength(a, b):
    """ Check two arrays are same lengths """
    if len(a) == len(b):
        return True
    else:
        return False 

def to_quantity(x, unit):
    """ Force x to be astropy.Quantity """
    if isinstance(x, Quantity):
        return x.to(unit)
    else:
        return Quantity(x, unit=unit)

def check_lengths(a, b):
    """ Check len(a) == len(b) and raise ValueError if not """
    # Quantity does not return True with np.isscalar, even if q.isscalar=True
    if not issamelength(a, b):
        raise ValueError(f"Lengths of DimensionScales do not match: {len(a)} vs {len(b)}")

class ArrayBasedDimensionScale(object):
    """ TODO """
    def __init__(self, name, np_array, units=None):
        self.name = name
        self.data = np_array
        self.units     = Unit(units) if isinstance(units, str) else units
        self.shape = np_array.shape
        self.ndim  = np_array.ndim
        self.dtype = np_array.dtype

    def __getitem__(self, i):
        """ Allow indexing with scale[i] """
        return self.data[i]

class DimensionScale(object):
    """ Dimension Scale 'duck array' with basic numpy/astropy array support 
    
    Creates an object maps an array index (i) to a dimension value. 
    The dimension scale supports evenly-spaced data than can be described as
    
        dim_scale_value = start_value + step_size * i
    
    Crucially, DimensionScale does not generate an actual array until 
    explicitly requested, so keeps memory usage low. 
    
    Calling `numpy.asarray(ds)` will evaluate the array and generate a numpy
    array. Similarly, `ds.generate()` will return an astropy.Quantity array.
    
    DimensionScale is designed to 'attach' to a numpy-like array to 
    serve as dimension metadata. For example, to represent a frequency scale
    from 1.0-1.1 GHz on a data array d:
    
        d = np.array([1,2,3,4])
        ds = DimensionScale('frequency', 1.0, 1.1, len(d), 'GHz')
    
    Indexing a DimensionScale object will return a new DimensionScale with
    updated start/stop/step values:
        
        ds = DimensionScale('frequency', 1000, 0.1, n_step=100, units='GHz')
        # Returns <DimensionScale 'frequency': start 1000 GHz step 0.1 GHz nstep 100 >
        
        ds_sub = ds[10:20:2]
        # Returns <DimensionScale 'frequency': start 1001.0 GHz step 0.2 GHz nstep 5 >
    
    DimensionScales can be added, subtracted, multiplied, and divided, propagating 
    astropy units (and raising errors if incompatible). Unit conversion is also 
    implemented, using `ds.to('new_unit')`. 
    
    Some usage examples:
    
        d = np.arange(0, 1000)
        ds = DimensionScale('frequency', 1.0, 1.1, len(d), 'GHz')
        
        # Do some indexing on d to get a subset of data
        d_sub  = d[10:100:2]
        ds_sub = ds[10:100:2]
        
        # generate numpy array
        ds_array = np.asarray(ds)
        # generate astropy.Quantity array
        ds_astropy = ds.generate()
        
        # Convert units
        ds_hz = ds.to('Hz')
    
    # Useful references 
        [1] https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html
        [2] https://numpy.org/neps/nep-0030-duck-array-protocol.html
        [3] https://numpy.org/neps/nep-0018-array-function-protocol.html
        [4] https://numpy.org/devdocs/user/basics.dispatch.html
    
    """
    def __init__(self, name, val_start, val_step, n_step, units=None):
        self.name = name
        if isinstance(val_start, Quantity):
            self.val_start = val_start.to(units).value
        else:
            self.val_start = val_start
        if isinstance(val_step, Quantity):
            self.val_step  = val_step.to(units).value
        else:
            self.val_step  = val_step
        self.n_step    = n_step
        self.units     = Unit(units) if isinstance(units, str) else units
        self.shape = (n_step,)
        self.ndim  = 1
        self.dtype = np.dtype('float64')
    
    @property
    def step(self):
        return Quantity(self.val_step, unit=self.units)
    
    @property
    def start(self):
        return Quantity(self.val_start, unit=self.units)

    def _generate_array(self, start_idx=None, stop_idx=None):
        """ Generate an numpy array from this DimensionScale object
        
        This is called by __array__() with no args for np.asarray support.
        """
        start_idx = 0 if start_idx is None else start_idx
        stop_idx = self.n_step if stop_idx is None else stop_idx
        return np.arange(start_idx, stop_idx) * self.val_step + self.val_start

    def __len__(self):
        """ Return length of array, called when len(x) is used on the object """
        return self.n_step
        
    def __repr__(self):
        return f"<DimensionScale '{self.name}': start {self.val_start} {self.units} step {self.val_step} {self.units} nstep {self.n_step} >"

    def __getitem__(self, i):
        """ Supports indexing of the DimensionScale, e.g. ds[0:10:2] """
        if isinstance(i, slice):
            skip  = 1 if i.step is None else i.step
            start = 0 if i.start is None else i.start
            stop  = self.n_step if i.stop is None else i.stop
            new_val_start = self.val_start + start * self.val_step
            new_n_step = (stop - start) // skip
            new_step_val = self.val_step * skip
            #return self.val_start + self.val_step * np.arange(start, stop, step)
            return DimensionScale(self.name, new_val_start, new_step_val, new_n_step, units=self.units)
        else:
            if i < 0 or i > self.n_step:
                raise IndexError(f"Index ({i}) is out of axis bound ({self.n_step})")
            return i * self.val_step + self.val_start
    
    def __array__(self):
        """ Returns an evaluated numpy array when np.asarray called. 
        
        See https://numpy.org/neps/nep-0030-duck-array-protocol.html
        """
        return self._generate_array()

    def __duckarray__(self):
        """ Returns itself (original object) 
        
        Note: proposed in NEP 30 but not yet implemented/supported.
        Idea is to return this when np.duckarray is called.
        See https://numpy.org/neps/nep-0030-duck-array-protocol.html
        """
        return self
    
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DimensionScale objects
        if not all(issubclass(t, DimensionScale) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    
    def __add__(self, other):
        """ Implement element-wise addition support with unit handling """
        if isscalar(other):
            other = to_quantity(other, self.units)
            new_val = (self.val_start + other.value) * self.units
            return DimensionScale(self.name, new_val.value, self.val_step, self.n_step, units=new_val.unit)
        else:
            check_lengths(self, other)
            other_val = (other.val_start * other.units).to(self.units).value
            other_val_step = (other.val_step * other.units).to(self.units).value
            new_val      = (self.val_start + other_val) * self.units
            new_val_step = (self.val_step + other_val_step) * self.units
            return DimensionScale(self.name, new_val.value, new_val_step.value, self.n_step, units=new_val.unit)

    def __sub__(self, other):
        """ Implement element-wise subtraction support with unit handling """
        if isscalar(other):
            other = to_quantity(other, self.units)
            new_val = (self.val_start - other.value) * self.units 
            return DimensionScale(self.name, new_val.value, self.val_step, self.n_step, units=new_val.unit)
        else:
            check_lengths(self, other)
            other_val = (other.val_start * other.units).to(self.units).value
            other_val_step = (other.val_step * other.units).to(self.units).value
            new_val      = (self.val_start - other_val) * self.units
            new_val_step = (self.val_step - other_val_step) * self.units
            return DimensionScale(self.name, new_val.value, new_val_step.value, self.n_step, units=new_val.unit)

    def __mul__(self, other):
        """ Implement element-wise muliplication support with unit handling """
        if isscalar(other):
            other = to_quantity(other, self.units)
            new_val = (self.val_start * other.value) * self.units
            return DimensionScale(self.name, new_val.value, self.val_step, self.n_step, units=new_val.unit)
        else:
            check_lengths(self, other)
            other_val = (other.val_start * other.units).to(self.units).value
            other_val_step = (other.val_step * other.units).to(self.units).value
            new_val      = (self.val_start * other_val) * self.units
            new_val_step = (self.val_step * other_val_step) * self.units
            return DimensionScale(self.name, new_val.value, new_val_step.value, self.n_step, units=new_val.unit)

    def __truediv__(self, other):
        """ Implement element-wise muliplication support with unit handling """
        if isscalar(other):
            other = to_quantity(other, self.units)
            new_val = (self.val_start / other.value) * self.units
            return DimensionScale(self.name, new_val.value, self.val_step, self.n_step, units=new_val.unit)
        else:
            check_lengths(self, other)
            other_val = (other.val_start * other.units).to(self.units).value
            other_val_step = (other.val_step * other.units).to(self.units).value
            new_val      = (self.val_start / other_val) * self.units
            new_val_step = (self.val_step / other_val_step) * self.units
            return DimensionScale(self.name, new_val.value, new_val_step.value, self.n_step, units=new_val.unit)
    
    def to(self, new_unit):
        """ Convert units (using astropy units) 
        
        Returns a copy of the DimensionScale with units converted to new_unit
        """
        current_val      = self.val_start * self.units
        current_val_step = self.val_step * self.units
        new_val = current_val.to(new_unit)
        new_val_step = current_val_step.to(new_unit)
        return DimensionScale(self.name, new_val.value, new_val_step.value, self.n_step, units=new_val.unit)
    
    def generate(self, start_idx=None, stop_idx=None):
        """ Generate an astropy.Quantity array from this DimensionScale object """
        return Quantity(self._generate_array(start_idx, stop_idx), unit=self.units)
    
    def index(self, val, val2=None):
        """ Lookup the closest index where DimensionScale equals value """
        #x = x0 + i * dx
        # i = (x - x0) / dx
        if isinstance(val, np.ndarray):
            i = ( (val - self.val_start) / self.val_step).astype('int32')
            if np.min(i) < 0:
                raise ValueError("one or more values fall outside DimensionScale range")
            return i
        else:
            i = round( (val - self.val_start) / self.val_step)
            if i < 0:
                    raise ValueError("value falls outside DimensionScale range")
            if val2 is None:
                return i
            else:
                j = round( (val2 - self.val_start) / self.val_step)
                if j < 0:
                    raise ValueError("value2 falls outside DimensionScale range")
                return i, j

class TimeScale(DimensionScale):
    """ Time Scale 'duck array' with basic numpy/astropy array support 
    
    This is a subclass of DimensionScale, but uses astropy.Time and TimeDelta 
    to better handle time slicing.
    """
    def __init__(self, name, time_start, time_step, n_step, time_format='unix', time_delta_format='sec'):
            
        time_start_unix = Time(time_start, format=time_format).to_value('unix')
        time_step_sec   = TimeDelta(time_step, format=time_delta_format).to_value('sec')
        super(TimeScale, self).__init__(name, time_start_unix, time_step_sec, n_step, units='s')
        self.time_format = time_format
        self.time_delta_format = time_delta_format
    
    def __repr__(self):
        t0 = self.time_start
        dt = self.time_delta
        return f"<TimeScale '{self.name}': start {t0} {self.time_format} step {dt} {self.time_delta_format} nstep {self.n_step} >"
    
    @property
    def time_start(self):
        t = Time(self.val_start, format='unix')
        t.format = self.time_format
        return t

    @property
    def time_delta(self):
        t = TimeDelta(self.val_step, format='sec')
        t.format = self.time_delta_format
        return t
    
    @property
    def step(self):
        return Quantity(self.val_step, unit='s')
    
    @property
    def elapsed(self):
        t = TimeDelta(self.val_step * self.n_step, format='sec').to('s')
        return t
    
    def generate(self, start_idx=None, stop_idx=None):
        """ Generate an astropy.Time array from this TimeScale object """
        t = Time(self._generate_array(start_idx, stop_idx), format='unix')
        t.format = self.time_format
        return t

    def __getitem__(self, i):
        """ Supports indexing of the TimeScale, e.g. ts[0:10:2] """
        if isinstance(i, slice):
            skip  = 1 if i.step is None else i.step
            start = 0 if i.start is None else i.start
            stop  = self.n_step if i.stop is None else i.stop
            new_n_step    = (stop - start) // skip
            
            new_time_unix  = self.val_start + start * self.val_step
            new_time_obj   = Time(new_time_unix, format='unix')
            new_time_obj.format = self.time_format
            
            new_delta_sec = self.val_step * skip
            new_delta_obj  = TimeDelta(new_delta_sec, format='sec')
            new_delta_obj.format = self.time_delta_format
            
            #return self.val_start + self.val_step * np.arange(start, stop, step)
            return TimeScale(self.name, new_time_obj.value, new_delta_obj.value, new_n_step, 
                             time_format=new_time_obj.format, time_delta_format=new_delta_obj.format)
        else:
            if i < 0 or i > self.n_step:
                raise IndexError(f"Index ({i}) is out of axis bound ({self.n_step})")
            t = Time(i * self.val_step + self.val_start, format='unix')
            t.format = self.time_format
            return t
