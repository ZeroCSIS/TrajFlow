# Data Format

Due to policy restrictions, we could not distribute real dataset. The included `toy_data/` folder is fully synthetic and is provided only to document the processed data format expected by the released code.

## Included Toy Data

Generate or refresh the toy dataset with:

```bash
python data/make_toy_data.py
```

The default output is:

- `data/toy_data/traj_segments.pkl`
- `data/toy_data/conditions.pkl`
- `data/toy_data/mesh_mapping_dict.pkl`
- `data/toy_data/traj_mean_std.txt`
- `data/toy_data/conditions_mean_std.txt`
- `data/toy_data/grid_meta.json`
- `data/processed_coeffs_Toy_rdp_k_10.npy`

These files are created from procedural curves in a dummy coordinate space.
They do not correspond to real users, real trips, real OD pairs, or real
locations.

## Processed Dataset Schema

`traj_segments.pkl`

- Python pickle containing a `numpy.ndarray`
- Shape: `(num_samples, trajectory_length, 2)`
- Axis order: `[latitude_like_value, longitude_like_value]`
- For the toy data these are synthetic numeric placeholders.

`conditions.pkl`

- Python pickle containing a `numpy.ndarray`
- Shape: `(num_samples, 9)`
- Columns:
  1. `departure`: 5-minute departure-time bucket in `[0, 287]`
  2. `total_dis`: standardized total distance
  3. `total_time`: standardized travel time
  4. `total_len`: standardized trajectory length
  5. `avg_dis`: standardized average inter-point distance
  6. `avg_speed`: standardized average speed
  7. `starting_location`: integer origin cell id
  8. `ending_location`: integer destination cell id
  9. `trans_mode`: integer transport mode

Transport mode ids follow the preprocessing convention:

- `0`: walk
- `1`: car
- `2`: bus
- `3`: train
- `4`: bike

`mesh_mapping_dict.pkl`

- Python pickle containing a dictionary `{grid_cell_value: integer_id}`
- `conditions[:, 6:8]` stores the integer ids.
- The loader inverts this mapping internally when reconstructing OD cells.

`traj_mean_std.txt`

- Text file with two mean/std pairs:
  - `lat_mean`, `lat_std`
  - `lon_mean`, `lon_std`

`conditions_mean_std.txt`

- Text file with five mean/std pairs for columns 2-6 of `conditions.pkl`:
  - `total_dis`
  - `total_time`
  - `total_len`
  - `avg_dis`
  - `avg_speed`

`processed_coeffs_Toy_rdp_k_10.npy`

- Cached toy trajectory coefficients for the default toy config.
- Shape: `(num_samples, 10, 2)`
- Included so the toy smoke test does not need to recompute coefficients.

## Using Authorized Real Data

If you have authorized access to your own trajectory data, convert
it into the same processed format above and point the config to that folder:

```yaml
data:
  dataset_folder: data/your_processed_dataset
```
