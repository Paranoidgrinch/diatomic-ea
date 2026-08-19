# DiatomicEA user guide

## 1. Start the application

Launch **DiatomicEA** from the Start menu on Windows or the application menu on Linux. The compute panel reports whether the scientific backend is ready.

## 2. Define a molecule

Enter the chemical symbols for atom A and atom B. DiatomicEA normalizes the formula and validates the elements before the calculation is added to the queue.

## 3. Calculation settings

Set:

- **Bond-length scan:** lower and upper limits for the initial geometry scan in angstrom.
- **Maximum spin (2S):** highest spin value considered in the neutral/anion state scan.
- **Workers:** number of worker processes. The GUI shows the detected CPU resources and a conservative recommendation.

The standard scientific method is fixed. The GUI exposes only molecule-specific settings that are intended to vary between calculations.

## 4. Queue

Select **Add to queue** to freeze the current settings for that molecule. Waiting calculations may be reordered or removed. Select **Start queue** to process one molecule at a time.

**Stop after current** finishes the active molecule safely and then stops the queue. A failed or interrupted calculation can be resumed or retried; already persisted raw single-point results are reused.

## 5. Progress

The current-calculation panel shows:

- calculation stage
- completed tasks
- percentage
- tasks per second
- estimated time remaining
- elapsed time for the current stage

The scientific calculation runs outside the GUI process, so the application remains responsive.

## 6. Results

A completed calculation shows:

- predicted electron affinity
- 80%, 90% and 95% prediction intervals
- functional half-range
- individual PBE, B3LYP, PBE0 and TPSSh electron-affinity values

Select **Open results folder** to access raw and final files.

## 7. Stored data

Windows production data are stored below `%LOCALAPPDATA%\DiatomicEA\production_runs`. On systems without `LOCALAPPDATA`, DiatomicEA uses `~/.diatomic-ea/production_runs`.

Each scientific run preserves raw task records and provenance so that the calculation can be inspected and resumed without silently recomputing successful tasks.
