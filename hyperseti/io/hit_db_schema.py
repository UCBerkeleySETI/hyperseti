import os
import yaml
import re
from pathlib import Path
from copy import deepcopy
from . import load_config

# READ SCHEMA INFO
HERE = Path(__file__).parent.absolute()
SCHEMA_YML_PATH = os.path.join(HERE, 'hit_db_schema.yml')
SCHEMA_DICT = load_config(SCHEMA_YML_PATH)
SCHEMA_DICT['_schema_path'] = SCHEMA_YML_PATH

def get_col_schema(name: str) -> dict:
    """ Read column information from schema 

    Args:
        name (str): Name of column to lookup

    Notes:
        This handles regex lookup for 'bX_{col_name}_cY'
    
    Returns:
        col_info (dict): Information about column, as read from schema
    """
    
    # Regex: does col start with b0_ ... bX_
    pat_beam = r'b(\d+)_(\w+)'
    beam_match = re.search(pat_beam, name)
    if beam_match:
        beam_id = beam_match.group(1)
        col_name = beam_match.group(2)
        
        # Regex: check if it is a poly coefficient (endswith _c0 ... _cX)
        pat_coeff = r'(\w+)_c(\d+)$'
        coeff_match = re.search(pat_coeff, beam_match.group(2))
        if coeff_match:
            col_name = coeff_match.group(1)
            coeff_id = int(coeff_match.group(2))
            
            # Replace placehold indexes with actual index values
            # deepcopy makes sure SCHEMA_DICT is not modified
            d = deepcopy(SCHEMA_DICT[f'bX_{col_name}_cY'])
            print(f'HERE: coefficient {coeff_id} (c{coeff_id})')
            print(f"HERE2: {d}")
            d['description'] = d['description'].replace('coefficient Y (cY)',
                                                        f'coefficient {coeff_id} (c{coeff_id})')
        else:
            d = deepcopy(SCHEMA_DICT[f'b{0}_{col_name}'])
        d['description'] = d['description'].replace('beam X (bX)', 
                                            f'beam {beam_id} (b{beam_id})')
    else:
        d = deepcopy(SCHEMA_DICT[name])
    return d