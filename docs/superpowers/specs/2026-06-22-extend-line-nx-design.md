# Extend Line Nx Design

**Date:** 2026-06-22
**Status:** Approved

## Overview

Replace the hardcoded `extend_line_2x()` with a parameterized `extend_line_nx(factor)` to make the blue line extension multiplier configurable.

## Requirements

- **R1:** Add a configurable constant `LINE_EXTEND_FACTOR` (default: 3)
- **R2:** Rename `extend_line_2x` to `extend_line_nx`, accepting a `factor` parameter
- **R3:** Update the single call site to pass `LINE_EXTEND_FACTOR`

## Changes

### File: `cube-detection-01.py`

**1. Add constant** (after HOUGH parameters, around line 37):

```python
LINE_EXTEND_FACTOR = 3     # Blue line extension multiplier
```

**2. Replace function** (lines 215-226):

```python
def extend_line_nx(p1, p2, factor):
    """
    Extend a line segment to `factor`x its original length, centered at midpoint.
    Returns the two new endpoints.
    """
    x1, y1 = p1
    x2, y2 = p2
    ext1 = (int(round(factor * x1 - (factor - 1) * x2)),
            int(round(factor * y1 - (factor - 1) * y2)))
    ext2 = (int(round(factor * x2 - (factor - 1) * x1)),
            int(round(factor * y2 - (factor - 1) * y1)))
    return ext1, ext2
```

**3. Update call site** (line ~381):

```python
ext = extend_line_nx((x1, y1), (x2, y2), LINE_EXTEND_FACTOR)
```

## Error Handling

- `factor < 1`: lines will shrink — valid math, no special handling needed
- `factor = 1`: returns original points — identity operation
