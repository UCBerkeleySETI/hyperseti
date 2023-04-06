from hyperseti.io.hit_db import HitDatabase, generate_metadata, get_col_schema
from hyperseti.io import from_h5, load_config
from hyperseti.hit_browser import HitBrowser


import pandas as pd
from pprint import pprint
import os
import h5py

import pytest

try:
    from .file_defs import synthetic_fil, test_fig_dir, voyager_h5, voyager_csv, voyager_yaml
except:
    from file_defs import synthetic_fil, test_fig_dir, voyager_h5, voyager_csv, voyager_yaml

def test_hit_db():
    try:
        db = HitDatabase('test_db.h5', mode='w')

        df = pd.read_csv(voyager_csv)
        print(df)

        db.add_obs('voyager', df)
        db.add_obs('voyager_dupe', df)
        db.add_obs('voyager_with_yaml', df, config=load_config(voyager_yaml))

        print(db.list_obs())

        df_in_db = db.get_obs('voyager')
        print(df_in_db)

        del(db)

        db = HitDatabase('test_db.h5', mode='r')
        print(db.list_obs())
        df_in_db = db.get_obs('voyager')
        print(df_in_db)
        
        pprint(db.get_obs_metadata('voyager'))

        df_in_db2 = db.get_obs('voyager_with_yaml')
        config_roundtrip = db.get_obs_config('voyager_with_yaml')

        pprint(config_roundtrip)

    finally:
        if os.path.exists('test_db.h5'):
            os.remove('test_db.h5')

def test_generate_metadata():
    pprint(generate_metadata())

def test_browser():
    try:
        # Create a db from a HitBrowser
        dbfn = 'test_browser.hitdb'
        darr = from_h5(voyager_h5)
        v = pd.read_csv(voyager_csv)
        hb = HitBrowser(darr, v)
        hb.to_db(dbfn, 'test_obs')

        # Load a HitBrowser from the db
        db = HitDatabase(dbfn)
        hb2 = db.browse_obs('test_obs')
        print(hb2)
        print(hb2.data_array)
        print(hb2.hit_table)

        # Test with manual data_array pairing
        hb3 = db.browse_obs('test_obs', data_array=darr)

        db.close()
        # catch exception
        with pytest.raises(FileNotFoundError):
            db = HitDatabase(dbfn, mode='r+')
            db.h5['test_obs'].attrs['input_filename'] = 'MISSING.DATA'
            hb4 = db.browse_obs('test_obs')
            print(hb4.data_array)
    finally:
        if os.path.exists(dbfn):
            os.remove(dbfn)

def test_col_schema():
    cs = get_col_schema('b101_gulp_poly_c9')
    print(cs)
    assert(cs['description'] == 'Polynomial coefficient 9 for poly fit (c9), beam 101 (b101)')

    with pytest.raises(KeyError):
        cs = get_col_schema('unknown_column')

def test_read_back_col_schema():
    dbfn = 'test_db.h5'
    try:
        # Create database
        db = HitDatabase(dbfn, mode='w')
        df = pd.read_csv(voyager_csv)
        print(df)
        db.add_obs('voyager', df)

        schema_out = db.get_obs_schema('voyager')
        pprint(schema_out)

        del(db)  # Close h5 file

        h = h5py.File('test_db.h5')
        print(list(h.items()))

        print(list(h['voyager/snr'].attrs.items()))
        print(list(h['voyager/drift_rate'].attrs.items()))
    finally:
        if os.path.exists(dbfn):
            os.remove(dbfn)

if __name__ == "__main__":
    test_hit_db()
    test_generate_metadata()
    test_browser()
    test_col_schema()
    test_read_back_col_schema()