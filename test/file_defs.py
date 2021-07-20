from os import path

here = path.dirname(path.abspath(__file__))

voyager_fil = path.join(here, 'test_data/Voyager1.single_coarse.fine_res.fil')
voyager_h5 = path.join(here, 'test_data/Voyager1.single_coarse.fine_res.h5')
voyager_h5_flipped = path.join(here, 'test_data/Voyager1.single_coarse.fine_res.flipped.h5')
synthetic_fil = path.join(here, 'test_data/synthetic.fil')
synthetic_h5 = path.join(here, 'test_data/synthetic.fil')

test_fig_dir = path.join(here, 'test_figs')
