# test_tube Notes

This file keeps project-specific `test_tube.stl` commands out of the main README.

## Upright Opening Under Palm

This recipe stands the tube up, starts near the opening, and uses thumb, index, and middle only with a fixed `3 + 3 + 3` contact split.

```bash
uv run python -u scripts/optimize_primitive.py \
  --mesh-path /home/haegu/minimal_graspqp/assets/objects/test_tube.stl \
  --mesh-scale 0.001 \
  --mesh-pitch-deg 90 \
  --batch-size 2 \
  --num-steps 300 \
  --num-contacts 9 \
  --mala-star \
  --contact-switch-probability 0.4 \
  --allowed-contact-links th,ff,mf \
  --equalize-contacts-across-links \
  --init-surface-axis z \
  --init-surface-side max \
  --init-surface-band-fraction 0.15 \
  --palm-down \
  --log-every 1 \
  --profile-every 1 \
  --output outputs/test_tube_upright_opening_under_palm.pt
```

Single-sample visualization:

```bash
uv run python scripts/visualize_optimization_result.py \
  --input outputs/test_tube_upright_opening_under_palm.pt \
  --sample-index 0
```

Batch visualization:

```bash
uv run python scripts/visualize_optimization_batch.py \
  --input outputs/test_tube_upright_opening_under_palm.pt \
  --spacing 0.3 \
  --row-spacing 0.4 \
  --port 8081
```

Then open `http://localhost:8081`.

## Coordinate Notes

- The raw `test_tube.stl` lies mostly along the object `x` axis.
- The opening is on the `max` end of that long axis in the original orientation.
- `--mesh-pitch-deg 90` rotates the tube upright so the opening is biased toward `z max`.
- `--mesh-yaw-deg 180` only flips the tube around the `z` axis; it does not stand it up.
