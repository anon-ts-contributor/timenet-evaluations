# timenet-evaluations

A comparative benchmark. It loads one Sleep-EDF dataset, writes it in three storage formats —
pandas/parquet, torch, and TimeF — and measures how long each takes to read back and how much disk
each occupies. The run produces one JSON result and a printed summary.

It is deliberately never run in continuous integration: timings from two machines are not
comparable, so a green CI run would say nothing about performance.

```console
$ uv run timenet-evaluations <directory holding the Sleep-EDF recordings>
```

See the repository README for setup.
