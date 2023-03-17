import pylab as plt
import cupy as cp


def _get_extent(data_array, xaxis, yaxis):
    """ Generate extents for imshow axis 
    
    Args:
        xaxis (str): 'channel', 'frequency' supported
        yaxis (str): 'driftrate', 'driftidx' or 'time_elapsed', 'timestep' supported
        metadata (dict): Metadata dictionary
    """
    data = data_array.data
    if xaxis == 'channel':
        ex_x0, ex_x1 = -data.shape[2] // 2, data.shape[2] // 2    
    elif xaxis == 'frequency':
        f = data_array.frequency
        f_step = f.step.to('Hz').value
        ex_x0, ex_x1 = f_step * -data.shape[2] // 2, f_step * data.shape[2] // 2 
        
    if yaxis == 'driftrate':
        dr = data_array.drift_rate
        ex_y0, ex_y1 = dr.data[-1], dr.data[0]
    elif yaxis == 'driftidx':
        ex_y0, ex_y1 = data.shape[0], 0
    
    if yaxis == 'timestep':
        ex_y0, ex_y1 = data.shape[0], 0
    elif yaxis == 'time_elapsed':
        ex_y0, ex_y1 = data_array.time.step.value * data.shape[0], 0
        
    return (ex_x0, ex_x1, ex_y0, ex_y1)


def _imshow(data_array, xaxis, yaxis, show_labels=True, show_colorbar=True, 
            beam_id=0, *args, **kwargs):
    """ Generalised imshow function
    
    Args:
        xaxis (str): 'channel', 'frequency' supported
        yaxis (str): 'driftrate', 'driftidx' or 'time_elapsed', 'timestep' supported
        metadata (dict): Metadata dictionary
        show_labels (bool): Show labels on axes
        show_colorbar (bool): Show colorbar
        beam_id (int): Which beam to plot
    
    Notes:
        Can use kwargs['apply_transform'] = np.log (or any other ufunc) to transform data
        before plotting
    """

    data = cp.asnumpy(data_array.data[:, beam_id]).squeeze()
    
    if 'apply_transform' in kwargs:
        transform = kwargs.pop('apply_transform')
        data = transform(data)

    plt.imshow(data, aspect='auto',
              extent=_get_extent(data_array, xaxis, yaxis), *args, **kwargs)

    if show_colorbar:
        plt.colorbar()
    
    if show_labels:
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)

        
 
def imshow_dedopp(data_array, xaxis='channel', yaxis='driftrate', *args, **kwargs):
    """ Do imshow for dedoppler data 
    
    Args:
        xaxis (str): 'channel', 'frequency' supported
        yaxis (str): 'driftrate', 'driftidx' or 'time_elapsed', 'timestep' supported
        metadata (dict): Metadata dictionary
        show_labels (bool): Show labels on axes
        show_colorbar (bool): Show colorbar
        
    """
    # TODO: Fix y-axis ticks for stepped (non uniform) plan
    _imshow(data_array, xaxis, yaxis, *args, **kwargs)

def imshow_waterfall(data_array, xaxis='channel', yaxis='timestep', *args, **kwargs):
    """ Do imshow for spectral data """
    _imshow(data_array, xaxis, yaxis, *args, **kwargs)
    
def overlay_hits(hits, xaxis='channel', yaxis='driftrate', marker='x', c='red'):
    if xaxis == 'channel':
        x = hits['channel_idx']
    elif xaxis == 'frequency':
        x = hits['f_start']
    
    if yaxis == 'driftrate':
        y = hits['drift_rate']
    elif yaxis == 'driftidx':
        y = hits['drift_idx']
    
    plt.scatter(x=x, y=y, marker=marker, c=c)
