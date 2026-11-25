# DualRng

DualRng runs controlled experiments on the two hardware random number generators inside a TrueRNGpro, on a Raspberry Pi with a touchscreen. It asks two questions:

1. **Intention.** Can one person shift the generators toward more 1s or more 0s when asked to?
2. **Connection.** When two people try to connect, do the two generators become more correlated with each other?

Every session follows a protocol that is fixed before any data arrives: randomized blocks, one statistic chosen in advance, and a permutation test for its p-value. Each session adds one row to a ledger. A report combines the ledger rows, keeping real sessions and sham (control) sessions separate. A multi-scale scan is still available, but it is clearly marked exploratory and never feeds the ledger.

---

## Setup

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, pyserial, pygame
pip install -r requirements-dev.txt      # adds pytest
```

The versions in `requirements.txt` are the ones used on the Pi. The code also runs on Python 3.9 with numpy 2.0, scipy 1.13, matplotlib 3.9 and pygame 2.6, which is how the test suite was run during development.

## Running sessions

### GUI (touchscreen)

```bash
python app.py --gui
python app.py --gui --block-seconds 20 --blocks-per-label 10
python app.py --gui --source sim            # no hardware needed
```

On the idle screen, choose **Intention** or **Connection** and turn **Sham** on or off. For intention, also pick the **Display**: `trace` (the default) or `wave`. Then tap **Start Session**. The line under the buttons shows the block plan and the total duration. The summary at the top shows the result of the last session.

During a session:

| Where | What |
|---|---|
| Top left | The cue for the current block: **AIM HIGH**, **AIM LOW**, **CONNECT** or **REST**. Between blocks it shows the next cue, dimmed. |
| Top right | Block progress (`Block 5/24`) |
| Middle | Live feedback: the **trace** or the **wave** (see "Displays" below) |
| Bottom right | The current block's z (see "Live feedback") |
| Bottom center | Circles for completed blocks that went the intended way (z > 1). The colour shows how strongly: purple > 1, orange > 1.645, yellow > 2.33, pink > 3.09. |
| Bottom left | **Stop**. Tap it twice within 3 seconds to end the session early. |

A sham session shows **CONTROL**, the block progress, and a dimmed wave that does not respond. It shows no cues and no feedback.

#### Displays

**Trace (intention, default).** During each AIM HIGH or AIM LOW block, a line starts at the left edge as the block begins and reaches the right edge as it ends.

- **What the line shows:** it is the block's *cumulative deviation*. Each read moves it up if that read had more 1s than the generators' usual lean (estimated from REST blocks), and down if fewer. Up always means more 1s.
- **The arrow** on the right shows the aim: up for AIM HIGH, down for AIM LOW.
- **The gray curves** show where chance alone keeps the line 95% of the time (±1.96 √reads). A line that runs outside them in the aimed direction is doing something chance rarely does.
- **Colours** follow the same palette as the wave. The line goes from amber through yellow and green to pale gold as the block moves further the aimed way.
- **Between blocks and in REST blocks,** only the centre line shows.

This is the classic display for RNG intention work, the same idea as the Global Consciousness Project's cumulative deviation graphs.

**Wave (connection, and optional for intention).** This is the original sine-wave visual. It grows calmer and brighter as the current block's measure moves the intended way. It suits group and connection sessions, where the goal is coherence rather than a direction.

The display is part of the protocol. It is saved in `meta.json` and the ledger (`feedback` column: `trace`, `wave`, `text` for headless runs, `none` for sham), and it changes the config hash. The report shows results by display whenever more than one display has been used, so sessions run differently are never pooled unknowingly.

Double-tap the screen to leave fullscreen, and press Escape to quit. **However the session ends** (it finishes, you tap Stop, you quit, the window closes, or the app gets Ctrl-C or SIGTERM), the session is stopped, analyzed and added to the ledger. If the program is killed or the Pi loses power, the data recorded up to that point is already on disk. That session is analyzed the next time the app starts.

### Headless

```bash
python app.py run --type intention
python app.py run --type connection --sham
python app.py run --type intention --source sim --sim-delta 0.0005   # simulated, with an injected effect
```

Cues print in the terminal as each block starts. Ctrl-C stops the session early, and the session is still saved and analyzed.

### Other commands

```bash
python app.py report                    # combined results from the ledger
python app.py report --complete-only    # leave out sessions that were stopped or interrupted
python app.py analyze --pending         # analyze sessions left without results (e.g. after a crash)
python app.py scan sessions/<id>        # EXPLORATORY scan of one session
python app.py scan --legacy data        # EXPLORATORY scan of the pre-2.0 session CSVs
python app.py probe                     # check what the device actually sends (see "Hardware")
```

The protocol options apply to `gui` and `run`: `--block-seconds` (default 30), `--gap-seconds` (default 3), `--blocks-per-label` (default 8 for intention, 12 for connection) and `--bits-per-read` (default 1024). The simulator options are `--sim-lean P1 P2`, `--sim-delta`, `--sim-rho` and `--sim-rate`.

---

## The protocol

### Blocks

A session is a series of blocks. Each block is preceded by a short gap.

| Session type | Labels | Default |
|---|---|---|
| Intention | AIM HIGH, AIM LOW, REST | 8 of each = 24 blocks x 30 s, ~13 min |
| Connection | CONNECT, REST | 12 of each = 24 blocks x 30 s, ~13 min |

Every label appears the same number of times. The order is shuffled with Python's `secrets` module, which uses the operating system's cryptographic random source. It is never drawn from the generators under test. The gap before each block shows the next cue, so the person can get ready, and no data from the gap is analyzed.

### What is fixed before the session starts

When a session starts, files are written in this order:

1. `meta.json` records the full protocol configuration and a **config hash**, before the recorder starts. The configuration includes the session type, real or sham, block length and count, bits per read, the statistic, the tail, the permutation settings and the analysis version.
2. `schedule.csv` records every block's label and its start and end times on the monotonic clock. It is written and fsynced before the first block begins.
3. `reads.bin` is the raw data. It is appended continuously by the recorder.

The analysis assigns each read to a block by comparing the read's timestamps with the schedule. A read counts only if it falls entirely inside a block, and only blocks that ran to completion are used.

### Sham sessions

A sham session runs the same kind of schedule with the same files and the same analysis, but shows no cues and gives no feedback. Run sham sessions unattended, interleaved with real ones (for example a sham after every real session, or overnight). Their combined result is the control: it shows what the full pipeline produces when nobody is trying.

---

## What each number means

### The pre-specified statistic (one per session type)

**Intention:** `p1(HIGH) - p1(LOW)`. This is the proportion of 1 bits in all AIM HIGH blocks minus the proportion in all AIM LOW blocks, with both generators pooled.

- Both generators lean slightly toward 0. They sit near p(1) = 0.4992, not 0.5. Because the statistic is a difference between two conditions that share the same lean, the lean cancels out. Nothing assumes 0.5.
- The test is **one-sided**: the prediction is HIGH > LOW.

**Connection:** `mean z1*z2 in CONNECT - mean z1*z2 in REST`.

- For each read, each generator's proportion of 1s is converted to a z score using the mean and standard deviation from the session's own REST blocks. So the lean and the actual spread of the hardware come from the data, not from theory.
- `z1*z2` is positive when the two generators deviate in the same direction at the same moment. Its mean in CONNECT minus its mean in REST measures extra coupling during CONNECT.
- The test is **two-sided**, because coherence could show up as positive or negative coupling.
- **Secondary:** NetVar, computed the same way. Per read, Stouffer Z = (z1 + z2)/√2 and NetVar = Z². The secondary statistic is mean NetVar in CONNECT minus REST, tested **one-sided** (higher in CONNECT), following the Global Consciousness Project convention. It is reported alongside the primary but is not the primary.

### The p-value: a permutation test over block labels

Because the labels were assigned to blocks at random, the null hypothesis ("labels make no difference") says any other assignment of the same labels to the same blocks was equally likely. The test computes the statistic for every reassignment, or 20,000 random ones when there are too many to enumerate. **p** is the fraction of reassignments that give a statistic at least as extreme as the one observed.

- This assumes nothing about bias, drift or the shape of the noise in the hardware. The only assumption is that the labels were randomized, which the protocol guarantees.
- For connection sessions, the REST-based z scaling is recalculated for each reassignment.

**The resolution is set by the number of blocks.** With 8 HIGH and 8 LOW blocks there are 12,870 ways to assign the labels, so the smallest possible p is 1/12,870, and a single session's z cannot exceed about 3.8, however strong an effect is. Connection sessions use 20,000 random reassignments, which gives a similar ceiling of about 3.9. Collecting more bits per block makes a session more sensitive to small effects. Adding blocks raises that ceiling.

### Session results (`results.json`, one ledger row)

| Field | Meaning |
|---|---|
| `statistic` | The pre-specified statistic above |
| `p` | Permutation p-value, using the pre-specified tail |
| `z` | The same result on a normal scale. Converting it back with the same tail gives exactly `p`. It is positive when the statistic is high. Sessions are combined with this z. |
| `z_parametric` | A textbook normal approximation, for reference only. It can be much larger than `z` when the effect is large, because the permutation z has the ceiling described above. |
| `lean_p1`, `lean_p2` | p(1) of each generator, estimated from REST blocks |
| `secondary_*` | Connection only: the NetVar statistic, z and p |
| `blocks_used` | Blocks that completed and contained data |
| `bits_collected` / `bits_analyzed` | All bits recorded, and the bits inside analyzed blocks |
| `status` | `complete`, `stopped` (ended early), `interrupted` (the program died), `recorder_failed`; suffixed `-no-result` if too few blocks completed for a test |
| `feedback` | What the participant saw: `trace`, `wave`, `text` (headless) or `none` (sham) |
| `config_hash` | Identifies the exact protocol and analysis version |

### Live feedback (display only)

The GUI shows the session's own measure on the current block:

- **Intention:** the z of the block's proportion of 1s, compared with the lean estimated from REST blocks so far (before the first REST block: all other data so far, never the block being shown). The number at bottom right is the trace's current height divided by √reads, so up is positive. The tally circles and colours are signed by the aim, so they always mean "the way you are aiming".
- **Connection:** the sum of `z1*z2` over the block divided by √(reads in the block).

Live feedback is not the result. The confirmatory p comes from the permutation test after the session.

### The report: combining sessions

`python app.py report` combines the ledger rows with Stouffer's method, Z = Σz/√k. It does this **separately for each session type and for real and sham sessions**, and uses each type's tail. It also shows the difference in mean z between real and sham sessions.

How to read it:

- **Sham Z** should look like chance (|Z| under 2 most of the time). If it doesn't, something in the hardware or pipeline is producing effects on its own, and real-session results can't be trusted until that's found.
- **Real Z** is the confirmatory result.
- **Real vs sham** tests whether real sessions differ from controls run through the same pipeline.

Every session that starts is recorded, including sessions that were stopped early or interrupted. Otherwise, stopping when things look bad would quietly bias the totals. `--complete-only` shows the totals without them. Decide in advance how many sessions you will run, so the stopping point for the series is fixed too.

### Exploratory scan (never in the ledger)

`exploratory.json` and the lower panel of `session.png` show a multi-scale scan. Its statistic is the largest standardized windowed sum of the session's measure over all window lengths (5–160 s) and all positions, using full windows only. Its null distribution comes from circular time shifts:

- **Connection:** generator 2 is shifted against generator 1. Before scanning, z1·z2 is centered on its session mean, in the data and in every shifted copy. This matters because this device has a small steady coupling between its generators (see "Hardware"). A coupling that lasts all session isn't a local event, and without centering, the shifts would remove it from the null while it stayed in the data. Ordinary sessions would then look unusual: in a test with a steady coupling, the uncentered scan flagged 20% of null sessions instead of 5%.
- **Intention:** the aim sequence is shifted against the data.

This keeps each stream's own bias, drift and autocorrelation and breaks only their alignment. `p_scan` answers: "is there a stretch somewhere in this session that is more unusual than chance would give, at any time scale?" Use it to generate hypotheses for future pre-specified tests, not as evidence.

---

## Files

```
sessions/
  ledger.csv                 one row per session (confirmatory only)
  <YYYYmmdd_HHMMSS>_<type>[_sham]/
    meta.json                frozen config, hash, status, timing
    schedule.csv             block labels and times, written before the first block
    reads.bin                raw recording (not committed to git by default; can be large)
    acq_log.csv              recorder start/stop, resyncs, errors
    results.json             confirmatory result
    exploratory.json         scan result (EXPLORATORY)
    session.png              per-block chart + exploratory trace
```

`reads.bin` starts with a 16-byte header. Each read then adds one fixed-size record per generator: the monotonic start and end time, a sequence number shared by the two generators of a read, the generator number, n, the count of ones, and the n LSBs packed into bytes.

Code layout: `dualrng/recorder.py` (acquisition), `sources.py` (hardware and simulator), `records.py` (file format), `protocol.py` (configuration and schedule), `stats.py` (statistics and permutation tests), `analysis.py` (finalization), `ledger.py` (ledger and report), `scan.py` (exploratory scan), `live.py` (live feedback), `session.py` (session runner), `probe.py` (device check), and `sine_gui.py` / `app.py` (UI and command line).

### Acquisition design

The recorder is a separate process. It reads the device continuously, with no sleeps and no mode switching, and appends every read to `reads.bin`. Each write is flushed immediately and fsynced every second. The GUI, live feedback and analysis only read that file, so none of them can stall acquisition. If the main program dies, the recorder notices within a second, releases the serial port and exits.

**One clock for both processes.** Block times are written by the main program, and read timestamps by the recorder, so the two must read the same clock. Both use `dualrng/clock.py`, the system-wide monotonic clock (`clock_gettime(CLOCK_MONOTONIC)`). Python's `time.monotonic()` is not safe here: on at least one macOS Python build, a spawned process's `time.monotonic()` disagreed with its parent's by seconds, which put reads in the wrong blocks. At every session start, the recorder reports its clock reading and the session refuses to start if the two processes disagree by more than half a second. The measured offset is saved in `meta.json` (`recorder_clock_offset_s`).

---

## Hardware

### What the documentation says, and what this code relies on

- **Mode selection** works by a baud-rate "knock": open the port at 110, 300 and 110 baud, then at the mode's baud rate. This is documented by ubld.it.
- **MODE_RAW_BIN (19200)** is the mode used here. Its documented format is 4-byte packets. Bits 7:6 of every byte are a sequence code: generator 1 high byte, generator 1 low byte, generator 2 high byte, generator 2 low byte. Together the bytes hold one 10-bit ADC sample from each generator. Pairing by packet is therefore guaranteed by the format. The parser checks every tag, resynchronizes on any misalignment, and logs the count in `acq_log.csv`. The bit analyzed is the LSB of each 10-bit sample, the same bit the earlier version used.

### Where the documentation was unclear, and the calls I made

1. **MODE_UNWHITENED (57600) is not documented.** It appears only in the vendor's example scripts, as "Unwhitened RNG1-RNG2 (TrueRNGproV2 Only)". That doesn't say whether the stream interleaves the two generators or carries their difference, nor whether it is ASCII or binary. The earlier code parsed it as ASCII and split it by alternating values, but that behaviour couldn't be verified. **Decision:** use the documented MODE_RAW_BIN instead.
2. **The command bytes 0xB0 (reset), 0xC4 and 0xC5 (select generator 1 or 2)** that the earlier code sent are not in any vendor document I found. The vendor page says modes are selected by baud rate precisely so the device does not have to listen for commands. **Decision:** don't send them.
3. **Simultaneity.** The RAW_BIN documentation shows one sample from each generator per packet, but it does not state that the two samples are taken at the same instant. Within a packet they are at least adjacent, far closer than the old sequential reads 0.2 s or more apart.
4. **LSB bias.** Vendor staff say the ADC LSB carries manufacturing-dependent bias. This fits the leans seen in practice: p(1) ≈ 0.4992 in the old data, and about 0.483 (generator 1) and 0.500–0.503 (generator 2) for the RAW_BIN LSBs on this unit (see below). **Decision:** keep the LSB, for continuity with the existing data, and handle the bias through the protocol: lean estimated from REST blocks, statistics that cancel it, and permutation p-values. If you prefer, the full 10-bit samples are available in the parser for a different bit choice later.
5. **Read timestamps.** Each read is timestamped when its bytes arrive. USB and driver buffering mean the bits were generated slightly earlier, by an unknown but small amount when reading continuously. The 3 s gap before each block absorbs this.
6. **RAW_ASC sample text and the TrueRNG V1/V2 differences** are inconsistent on the vendor pages. They aren't used here.

### What the probe found on this unit

`python app.py probe` captures a sample in MODE_UNWHITENED and in MODE_RAW_BIN, saves the raw bytes under `probe/`, and prints what it finds. Run it again after changing hardware. On this TrueRNGproV2 (November 2026, 100,000 packets):

- **MODE_RAW_BIN works exactly as documented.** There were 0 resyncs and 0 discarded bytes. It runs at about 117,000 samples per second per generator, which is about 114–118 paired reads per second at the default 1,024 bits per read. 10-bit means are 525 and 524, with SDs of 81 and 75, within the vendor's range.
- **The LSB leans differ by generator.** Generator 1 sits at p(1) ≈ 0.483, about 10 standard errors from 0.5, and it was the same in a later session (0.484 from REST blocks). Generator 2 is close to 0.5 (0.503 in the probe, 0.500 in the session). The statistics handle this by design; see "LSB bias" above.
- **The generators are slightly coupled.** The correlation between the two generators' LSBs is r ≈ −0.011, about 3.4 standard errors from zero. This is plausibly crosstalk, since the two samples are taken next to each other in each packet. A steady coupling cancels in the confirmatory connection statistic (it is present in both CONNECT and REST). The exploratory scan centers it out. Sham sessions will measure it more precisely over time.
- **MODE_UNWHITENED** is ASCII, comma-separated, with values from 119 to 392 centered on 256 and no minus signs. Even and odd positions are statistically identical, so there's no evidence it alternates two generators, though identical generators would look the same. A centre of 256 with no negatives hints at a difference of the two generators shifted by 256, but the spread is too small for a plain difference of the RAW_BIN samples. Its format is still undetermined; it isn't used.

### Other judgment calls

- The tails are fixed in code: intention is one-sided (HIGH > LOW), connection is two-sided, and NetVar is one-sided (higher in CONNECT).
- Stopped and interrupted sessions are included in the ledger and the report by default, flagged, with `--complete-only` as the alternative.
- The connection z scaling uses the empirical mean and SD of REST reads, not the binomial SD, so any excess hardware variance is handled.
- The old baseline collection is replaced by sham sessions. Sham sessions are the control for everything in the pipeline, not just the generators.

---

## Validation

`pytest` runs the following, using the simulated source (no hardware needed):

- **Null calibration:** 400 simulated sessions of each type give uniform p-values (KS test, and a rejection rate at 0.05 within binomial limits). This holds both with p(1) = 0.5 and with the hardware's lean (0.4992 / 0.4994). The NetVar secondary is checked too.
- **Power:** with an injected intention effect of 2.5 standard errors (HIGH +δ, LOW −δ) and with an injected shared signal (correlation 0.06 in CONNECT blocks), the detection rate over 300 sessions matches the power expected for that effect size within ±0.09.
- **Exact enumeration and Monte Carlo agree**, the ledger z round-trips to the session p, and stopped sessions use only completed blocks.
- **The exploratory scan** is calibrated under the null (rejection rate within binomial limits), stays calibrated when the generators have a steady coupling like this unit's, and locates an injected local burst.
- **Acquisition:** RAW_BIN parsing, including chunking, garbage bytes and a dropped byte. Also the file format, including a truncated last record, and the recorder process.
- **End to end:**
  - A headless session completes and is saved.
  - A strong injected HIGH/LOW effect, run through the real recorder process, shows up in the right blocks with the right sign.
  - The recorder's timestamps match the main program's clock, including for a recorder started after an earlier one.
  - The real GUI loop, driven by taps, produces a finalized session and a ledger row when you tap Stop and when you close the window mid-session.
  - An intention session with the trace display draws the trace during aimed blocks and records `feedback: trace`.
  - A session killed with SIGKILL is recovered on the next start.

The simulator draws bits with `p(1) = lean ± δ` for intention effects. It adds a shared random shift to both generators' p(1) to give a stated correlation `rho` between their per-read z scores. Use `--sim-delta` and `--sim-rho` to see what an effect of a given size looks like in the GUI and report.

---

## The pre-2.0 data in `data/`

The earlier version had problems that the review of the code and the February 2026 data found:

- Its z scores assumed p(1) = 0.5.
- About two thirds of each session's wall time was not sampled, because collection stalled while charts were redrawn.
- The two generators were read one after the other, not as pairs.
- Its multi-scale analysis tested half-full windows as if full, and its 8-second scale correlated only 2–4 points.
- Its Global Z treated overlapping windows as independent.
- Its event CSVs were re-written on every refresh. The 13 `multiscale_events_*.csv` files hold 13,593 rows, but only 353 unique (time, scale) events. 176 of those are at the 5-minute scale, mostly from half-full windows early in a session.

Re-running those sessions through the new exploratory connection scan (`python app.py scan --legacy data`, saved in `reports/legacy_exploratory_scan.txt`):

- **None of the 17 sessions has p_scan < 0.05.** The smallest is 0.075, and about 0.9 would be expected below 0.05 by chance.
- Generator leans are 0.4990–0.4995 in every session.
- The correlation between z1 and z2 per read ranges from −0.058 to +0.065.
- The no-stall fraction of wall time is about 40%.

This is exploratory. The legacy sessions had no randomized protocol, so they can't be analyzed with the confirmatory statistics.

---

## References

- Stouffer, S. A. et al. (1949). *The American Soldier*. Princeton University Press. (Stouffer's method for combining z scores.)
- Nelson, R. D. et al. (2002). Correlations of continuous random data with major world events. *Foundations of Physics Letters*, 15(6), 537–550. (NetVar as used by the Global Consciousness Project.)
- Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses*. Springer. (Permutation tests based on randomization.)
- Kulldorff, M. (1997). A spatial scan statistic. *Communications in Statistics – Theory and Methods*, 26(6), 1481–1496. (Scan statistics, max over windows.)
- ubld.it: [How to change TrueRNGpro's mode](https://ubld.it/products/about/how-to-change-truerngpros-mode), [TrueRNGpro raw binary format](https://ubld.it/products/support/truerngpro-raw-binary-format); vendor example scripts at [github.com/euler357/TrueRNG](https://github.com/euler357/TrueRNG).
