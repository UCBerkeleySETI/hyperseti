## Hyperfrost

Hyperfrost is a version of hyperseti that can be used in [bifrost](https://github.com/ledatelescope/bifrost) pipelines. 



Under the `hyperfrost` section in YAML file, 

### Options

* `input_path` - Path to input directory
* `output_path` - Path to output directory
* `file_ext` - Search pattern for file extension, e.g. `*.0000.h5`
* `db_prefix` - Name for HitDatabase will be {db_prefix}_{timestamp}.hitdb
* `gulp_size`- Number of frequency channels to process at once.

Optional:

* `n_workers` - Number of worker threads (default 1). Increasing this may improve performance, at the expense of GPU usage. If reading data from disk is the bottleneck, leave this equal to 1 as larger values will give a speedup. 
* `gulp_overlap` - Number of channels to overlap when reading gulps (default 0). 


### Example config (YAML)

```yaml
hyperfrost:
  input_path: '/datax2/gbt/voyager'
  output_path: './output'
  file_ext: '*.0000.h5'
  db_prefix: 'gbt_voyager'
  gulp_size: 524288              # 2^19
  gulp_overlap: 0
  n_workers: 1
```


