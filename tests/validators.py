import xarray as xr


class DatasetValidator:
    def __init__(self, ds: xr.Dataset):
        self.ds = ds.cf
        self.issues = []

    def _validate_coord(self, cf_key, expected_name, expected_range, check_monotonic):
        # if any check fails, the total status will be False, but it won't block the issues from being populated
        success = True

        try:
            coord = self.ds[cf_key]
            coord_name = coord.name
        except KeyError:
            self.issues.append(f"no {expected_name} coord' {cf_key}' found")
            return False

        if coord_name != expected_name:
            self.issues.append(
                f"{expected_name} name is '{coord_name}', expected '{expected_name}'"
            )

        coord_min = float(coord.min())
        coord_max = float(coord.max())

        if coord_min < expected_range[0] or coord_max > expected_range[1]:
            self.issues.append(
                f"{expected_name} range [{coord_min}, {coord_max}] outside {expected_range}"
            )
            success = False

        if check_monotonic:
            is_increasing = (coord.diff(coord.name) > 0).all().item()

            if not is_increasing:
                self.issues.append(f"{expected_name} is not monotonically increasing")
                success = False

        return success

    def validate_lon(self, expected_range=(-180, 180), check_monotonic=False):
        return self._validate_coord("lon", "lon", expected_range, check_monotonic)

    def validate_lat(self, expected_range=(-90, 90), check_monotonic=False):
        return self._validate_coord("lat", "lat", expected_range, check_monotonic)

    def get_issues(self):
        return self.issues
