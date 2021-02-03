import pylab as plt

def get_extent(data, metadata, xaxis, yaxis):
    """ Generate extents for imshow axis 
    
    Args:
        xaxis (str): 'channel', 'frequency' supported
        yaxis (str): 'driftrate', 'driftidx' or 'time_elapsed', 'timestep' supported
        metadata (dict): Metadata dictionary
    """
    if xaxis == 'channel':
        ex_x0, ex_x1 = 0, data.shape[1]    
    elif xaxis == 'frequency':
        ex_x0, ex_x1 = metadata['fch1'], metadata['fch1'] + metadata['df'] * data.shape[1]
        
    if yaxis == 'driftrate':
        ex_y0, ex_y1 = metadata['drift_trials'][-1], metadata['drift_trials'][0]
    elif yaxis == 'driftidx':
        ex_y0, ex_y1 = data.shape[0], 0
    
    if yaxis == 'timestep':
        ex_y0, ex_y1 = data.shape[0], 0
    elif yaxis == 'time_elapsed':
        ex_y0, ex_y1 = metadata['dt'] * data.shape[0], 0
        
    return (ex_x0, ex_x1, ex_y0, ex_y1)


def _imshow(data, metadata, xaxis, yaxis, show_labels=True, show_colorbar=True, *args, **kwargs):
    plt.imshow(data, aspect='auto',
              extent=get_extent(data, metadata, xaxis, yaxis), *args, **kwargs)

    if show_colorbar:
        plt.colorbar()
    
    if show_labels:
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)

        
def imshow_dedopp(dedopp, metadata, xaxis='channel', yaxis='driftrate', *args, **kwargs):
    """ Do imshow for dedoppler data """
    _imshow(dedopp, metadata, xaxis, yaxis, *args, **kwargs)

    
def imshow_waterfall(data, metadata, xaxis='channel', yaxis='timestep', *args, **kwargs):
    """ Do imshow for spectral data """
    _imshow(data, metadata, xaxis, yaxis, *args, **kwargs)

def overlay_hits(hits, xaxis='channel', yaxis='driftrate', marker='x', c='red'):
    if xaxis == 'channel':
        x = hits['channel_idx']
    elif xaxis == 'frequency':
        x = hits['f_start']
    
    if yaxis == 'driftrate':
        y = hits['driftrate']
    elif yaxis == 'driftidx':
        y = hits['drift_idx']
    
    plt.scatter(x=x, y=y, marker=marker, c=c)
