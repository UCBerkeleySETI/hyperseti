from hyperseti.io.hit_db import HitDatabase

import pandas as pd

try:
    from .file_defs import synthetic_fil, test_fig_dir, voyager_h5, voyager_csv
except:
    from file_defs import synthetic_fil, test_fig_dir, voyager_h5, voyager_csv

def test_hit_db():
    db = HitDatabase('test_db.h5', mode='w')

    df = pd.read_csv(voyager_csv)
    print(df)

    db.add_obs('voyager', df)
    db.add_obs('voyager_dupe', df)

    print(db.list_obs())

    df_in_db = db.get_obs('voyager')
    print(df_in_db)

    del(db)

    db = HitDatabase('test_db.h5', mode='r')
    print(db.list_obs())
    df_in_db = db.get_obs('voyager')
    print(df_in_db)


if __name__ == "__main__":
    test_hit_db()
