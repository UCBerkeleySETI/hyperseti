## Investigating hits with HitBrowser


Results from `find_et` are returned as a `HitBrowser`, which has a `view_hits()` method for viewing this, and an `extract_hits()` method for extracting hits. 

A Pandas DataFrame of all hits found is attached as `hit_brower.hit_table`, and the data are accessible via `hit_browser.data_array`. 

```python
hit_browser = find_et(voyager_h5, config, gulp_size=2**20)
display(hit_browser.hit_table)

hit_browser.view_hit(0, padding=128, plot='dual')
```

![image](https://user-images.githubusercontent.com/713251/227728999-1bec6e2f-bfca-4ab7-ae59-d08010ad8a8d.png)
