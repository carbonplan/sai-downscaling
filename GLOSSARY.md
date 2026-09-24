# Glossary

> [!NOTE]
> This glossary was drafted with AI assistance (Claude Code)

The access data pages use these storage terms in the same sense as the Icechunk, Zarr, and
xarray documentation. Where this dataset uses a term more narrowly, the definition says how. For
terms that come up in the access utilities, such as lazy loading and data read, see the utilities'
[glossary](https://github.com/carbonplan/sai-downscaling-data-utils/blob/main/GLOSSARY.md).

**array**
A block of values laid out along named dimensions, like a table extended to more than two
dimensions. In this dataset, `tas` is an array on `(time, lat, lon)`, and each quality flag is an
array too. Arrays hold the actual numbers, and xarray shows each one as a variable when you open a
group. Zarr splits every array into chunks for storage.

**branch**
A named version of a repository. We publish each release of this dataset as a branch named
after the release, as of now only `v1.0.0`, so you can keep reading the same release even after a
newer one comes out. When you open a repository, always choose a branch by name. The `main`
branch of an output store exists but holds no data.

**chunk**
A fixed-size block of an array, compressed and stored on its own. Splitting arrays into
chunks means you can read just the part of the data you need, instead of the whole dataset.
Because a chunk is always read in full, the number of chunks a request touches sets how much
data it moves. In the downscaled product, one chunk covers 1 year over a 9° × 18° tile, about
3.8 MB before compression.

**group**
A named container for arrays and other groups, much like a folder that holds files
and subfolders. Groups let a single store hold many datasets, so you can open just the one
you need. In the output stores, the path `bcsd/ssp245/tas/003` points to one group, and opening it
with xarray gives you a dataset with that variable, its coordinates, and any quality flags.

**Icechunk**
An open-source storage engine for Zarr data that adds version control, similar to how Git
tracks changes to code. Every change is saved as a snapshot, and branches give
names to the versions you can open. We use Icechunk so that you can open any release by name. See
the [Icechunk documentation](https://icechunk.io/) to learn more.

**repository**
Icechunk's word for a store that also keeps a history of its versions. In these docs,
"store" and "repository" refer to the same thing: each GCM's output is one repository, and so is
each GCM's input. To read data, you open the repository and then choose one of its
branches, as the example under Data location in the published documentation shows.

**shard**
A bundle of chunks saved together as a single file in cloud storage, so the store
holds fewer, larger files. You can mostly ignore shards when you read data. A request still reads
only the chunks it needs, so shards don't change how much it moves. A downscaled shard holds 75
chunks, covering 3 years over a 45° × 90° tile.

**store**
The container that holds a whole tree of groups and arrays, along with their metadata. A
store takes the place that a single file has in formats like netCDF, but its contents are spread
across many smaller files in cloud storage. We publish one store per GCM for the output and one
per GCM for the input. In code, `session.store` is the store you pass to xarray or zarr-python.

**tree**
The way groups nest inside a store, like folders within folders. Zarr
calls this a hierarchy. Each part of a group's path is one level of the tree: in
`bcsd/ssp245/tas/003`, the levels are the method, scenario, variable, and ensemble member. xarray
can open a whole tree as a `DataTree`, but when you only need one dataset, opening its group
directly is quicker.

**Zarr**
An open, cloud-friendly format for large multidimensional arrays. Instead of
saving everything in one big file, Zarr stores data as many chunks organized into
a tree of groups, so tools can read just the pieces they need over the
internet. The stores in this dataset use Zarr version 3, which xarray and zarr-python can read.
