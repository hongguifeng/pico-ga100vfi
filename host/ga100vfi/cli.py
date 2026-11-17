"""Command line: classify, baseline, attempt, campaign, revert."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime

from .adaptive import Planner
from .bar0 import DEFAULT_BDF, Bar0, brd_serial, restore
from .classify import classify, snapshot
from .glitch import Attempt, Card, RecoveryError


def cmd_classify(args) -> int:
    if not restore(args.bdf, wait_s=0.5, settle_s=0.0):
        print(f"{args.bdf}: GPU config space unreachable (0xffffffff)", file=sys.stderr)
        return 2
    with Bar0(args.bdf) as bar:
        verdict = classify(bar)
    print(json.dumps(verdict.as_dict(), indent=2))
    return 0 if verdict.passed else 1


def cmd_baseline(args) -> int:
    if not restore(args.bdf, wait_s=0.5, settle_s=0.0):
        print(f"{args.bdf}: GPU config space unreachable (0xffffffff)", file=sys.stderr)
        return 2
    with Bar0(args.bdf) as bar:
        print(json.dumps(snapshot(bar), indent=2))
    return 0


def cmd_attempt(args) -> int:
    a = Attempt(args.v_dip, args.t_dip, args.v_restore,
                args.t_restore, args.t_revert, args.fast, args.hold_ms,
                args.pulses, args.pw, args.gap, args.delay, args.revert)
    try:
        with Card(args.port) as card:
            if not args.no_unlock and not _ensure_unlocked(card):
                print("unlock failed: controller locked until VDD33 cycles",
                      file=sys.stderr)
                return 2
            rep = card.attempt(a)
            print(f"pwd={rep.pwd} e1=0x{rep.e1:04x} nvvdd={rep.nvvdd} "
                  f"writes_failed={rep.writes_failed} "
                  f"recover={rep.recover}")
            if args.trace:
                print(card.trace())
    except RecoveryError as e:
        print(f"recovery failed: {e}", file=sys.stderr)
        return 2
    return 0


def _ensure_unlocked(card: Card) -> bool:
    pwd, _ = card.lock_status()
    if pwd:
        return True
    return card.unlock()


def _classify_attempt(v) -> str:
    """Classify a glitch attempt for the adaptive planner.

    under: dsov=1 (dip too shallow). over: dsov=0 but crc != 0x55 (too deep).
    hit: dsov=0 and crc=0x55 (in the window). goldilocks: all checks pass.
    """
    if v.passed:
        return "goldilocks"
    if v.dsov == 1:
        return "under"
    if v.crc_status == 0x55:
        return "hit"
    return "over"


def cmd_campaign(args) -> int:
    planner = Planner(args.min_v, args.max_v) if args.adaptive else None
    v_dip = args.v_dip
    spots: dict[int, dict[str, int]] = {}  # adaptive: v_dip -> outcome counts
    t0 = time.monotonic()
    tries = valid = lost = nack = dsov0 = 0
    verdict = None

    with Card(args.port) as card:
        if not args.no_unlock and not _ensure_unlocked(card):
            print("unlock failed: controller locked until VDD33 cycles",
                  file=sys.stderr)
            return 2
        try:
            while args.budget <= 0 or tries < args.budget:
                tries += 1
                if planner:
                    v_dip = planner.next_mv()
                a = Attempt(
                    v_dip_mv=v_dip, t_dip_us=args.t_dip,
                    v_restore_mv=args.v_restore, t_restore_us=args.t_restore,
                    t_revert_us=args.t_revert, fast=1, hold_ms=0,
                    pulses=args.pulses, pw=args.pw, gap=args.gap,
                    delay=args.delay, revert=0,
                )
                try:
                    rep = card.attempt(a)
                except RecoveryError as e:
                    print(f"[{tries}] recovery failed, stopping: {e}",
                          file=sys.stderr)
                    return 2
                except IOError as e:
                    nack += 1
                    print(f"[{tries}] glitch err: {e}", file=sys.stderr)
                    time.sleep(0.5)
                    if not args.no_unlock:
                        try:
                            _ensure_unlocked(card)
                        except IOError:
                            pass
                    continue
                if rep.writes_failed > 0:
                    nack += 1
                    if not args.no_unlock:
                        try:
                            _ensure_unlocked(card)
                        except IOError:
                            pass
                    continue
                outcome = "lost"
                if restore(args.bdf, wait_s=args.restore_wait):
                    with Bar0(args.bdf) as bar:
                        if bar.rd32(0x00) != 0xFFFFFFFF:
                            v = classify(bar)
                            if ((v.dsov >> 16) == 0xBADF or
                                    (v.crc_status >> 16) == 0xBADF):
                                print(f"[{tries}] fuse controller poison "
                                      f"(dsov=0x{v.dsov:08x} "
                                      f"crc=0x{v.crc_status:08x}); cold boot required",
                                      file=sys.stderr, flush=True)
                                return 2
                            outcome = _classify_attempt(v)
                if outcome == "lost":
                    lost += 1
                else:
                    valid += 1
                    if v.dsov == 0:
                        dsov0 += 1
                    print(f"[{tries}] v_dip={v_dip} valid={valid} lost={lost} dsov={v.dsov} "
                          f"crc=0x{v.crc_status:02x} priv=0x{v.priv_mask:08x} "
                          f"{'GOLDILOCKS' if v.passed else ''}",
                          file=sys.stderr, flush=True)

                if planner:
                    counts = spots.setdefault(v_dip, dict.fromkeys(
                        ("under", "hit", "over", "lost", "goldilocks"), 0))
                    counts[outcome] += 1
                if outcome == "goldilocks":
                    verdict = v
                    break
                if planner:
                    # A lost link is a too-deep outcome: the planner reads it as over.
                    planner.record(v_dip, outcome)
        finally:
            if not args.no_revert:
                try:
                    card.revert()
                except Exception as e:
                    print(f"warn: revert failed: {e}", file=sys.stderr)

    out = {
        "goldilocks": verdict is not None,
        "v_dip": v_dip,
        "attempts": tries,
        "valid": valid,
        "lost": lost,
        "nack": nack,
        "dsov_zero": dsov0,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "verdict": verdict.as_dict() if verdict else None,
    }
    if args.adaptive:
        out["spots"] = spots
    print(json.dumps(out, indent=2))
    if verdict is None:
        return 1
    with Bar0(args.bdf) as bar:
        serial = brd_serial(bar)
    entry = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "serial": serial,
        "bdf": args.bdf,
        "adaptive": args.adaptive,
        "attempts": tries,
        "profile": asdict(a),
    }
    try:
        with open(args.log, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        # Goldilocks is already reached; a log failure must not turn it into an error.
        print(f"warn: could not append to {args.log}: {e}", file=sys.stderr)
    return 0


def cmd_revert(args) -> int:
    with Card(args.port) as card:
        print(card.revert())
        print(card.status())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ga100vfi")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="run the five checks")
    c.add_argument("--bdf", default=DEFAULT_BDF)
    c.set_defaults(fn=cmd_classify)

    b = sub.add_parser("baseline", help="dump the shadow fuse registers")
    b.add_argument("--bdf", default=DEFAULT_BDF)
    b.set_defaults(fn=cmd_baseline)

    a = sub.add_parser("attempt", help="issue one glitch")
    a.add_argument("--port", default="/dev/ttyACM0")
    a.add_argument("--v-dip", type=int, default=325)
    a.add_argument("--t-dip", type=int, default=-100)
    a.add_argument("--v-restore", type=int, default=650)
    a.add_argument("--t-restore", type=int, default=-20)
    a.add_argument("--t-revert", type=int, default=0)
    a.add_argument("--delay", type=int, default=75,
                   help="us to advance the dip; rail responds ~75us after the write")
    a.add_argument("--pulses", type=int, default=1,
                   help="3 for the Goldilocks triple pulse")
    a.add_argument("--pw", type=int, default=0, help="pulse width us (0=auto)")
    a.add_argument("--gap", type=int, default=0, help="gap between pulses us (0=auto)")
    a.add_argument("--fast", type=int, default=1, choices=(0, 1),
                   help="MP2888B fast transient profile before the dip")
    a.add_argument("--hold-ms", type=int, default=0,
                   help="SCL hold after the train (0 = none)")
    a.add_argument("--revert", type=int, default=1, choices=(0, 1),
                   help="1=hand back to PWM-VID after dip; 0=stay PMBus (fast campaigns)")
    a.add_argument("--no-unlock", action="store_true")
    a.add_argument("--trace", action="store_true")
    a.set_defaults(fn=cmd_attempt)

    camp = sub.add_parser("campaign",
                          help="loop glitches until Goldilocks or the budget runs out")
    camp.add_argument("--port", default="/dev/ttyACM0")
    camp.add_argument("--bdf", default=DEFAULT_BDF)
    camp.add_argument("--budget", "--max-attempts", dest="budget", type=int,
                      default=500,
                      help="total attempts (0 = unlimited); --max-attempts is the "
                           "previous name for this flag")
    camp.add_argument("--log", default="goldilocks.jsonl",
                      help="on Goldilocks, append time, card serial, and the glitch "
                           "profile as one JSON line to this file")
    camp.add_argument("--v-dip", type=int, default=354,
                      help="dip voltage in mV (ignored with --adaptive)")
    camp.add_argument("--t-dip", type=int, default=-60)
    camp.add_argument("--v-restore", type=int, default=750)
    camp.add_argument("--t-restore", type=int, default=-20)
    camp.add_argument("--t-revert", type=int, default=0)
    camp.add_argument("--pulses", type=int, default=3)
    camp.add_argument("--pw", type=int, default=198)
    camp.add_argument("--gap", type=int, default=150)
    camp.add_argument("--delay", type=int, default=117)
    camp.add_argument("--restore-wait", type=float, default=0.25,
                      help="seconds to wait for PCIe link after PEX_RST")
    camp.add_argument("--no-unlock", action="store_true")
    camp.add_argument("--no-revert", action="store_true",
                      help="leave VRM in PMBus mode after campaign")
    camp.add_argument("--adaptive", action="store_true",
                      help="pick v_dip each shot by Thompson sampling over a "
                           "model fitted to the under/hit/over outcomes so far")
    camp.add_argument("--min-v", type=int, default=300,
                      help="adaptive: safety floor (mV)")
    camp.add_argument("--max-v", type=int, default=380,
                      help="adaptive: safety ceiling (mV)")
    camp.set_defaults(fn=cmd_campaign)

    r = sub.add_parser("revert", help="restore the stock control path")
    r.add_argument("--port", default="/dev/ttyACM0")
    r.set_defaults(fn=cmd_revert)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
