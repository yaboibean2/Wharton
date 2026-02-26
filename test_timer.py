#!/usr/bin/env python3
"""Simulate the countdown timer algorithm with realistic step completions.

Usage:
    python3 test_timer.py                # normal speed run
    python3 test_timer.py --fast         # everything 40% faster
    python3 test_timer.py --slow         # everything 60% slower
    python3 test_timer.py --mixed        # data slow, agents fast
    python3 test_timer.py --cached       # data cached (5s), agents normal
    python3 test_timer.py --all          # all scenarios
"""

import sys, math

_lp = {
    'data_gather': 47.9, 'agents': 18.1, 'blend': 0.001,
    'total': 65.3, 'avg_total': 63.0,
    'fundamentals': 40.0, 'price_history': 5.0, 'benchmark': 0.5,
    'value_agent': 12.0, 'growth_momentum_agent': 12.0,
    'macro_regime_agent': 12.0, 'risk_agent': 12.0, 'sentiment_agent': 15.0,
}


def build_scenario(mode='normal'):
    if mode == 'fast':
        s = 0.6
    elif mode == 'slow':
        s = 1.6
    elif mode == 'mixed':
        sd, sa = 1.5, 0.5
        de = 45.0 * sd  # 67.5
        return [
            (0.5*sd, 'benchmark'), (4.0*sd, 'price_history'), (de, 'fundamentals'),
            (de+6*sa, 'risk_agent'), (de+7*sa, 'value_agent'),
            (de+7.5*sa, 'growth_momentum_agent'), (de+8*sa, 'macro_regime_agent'),
            (de+9*sa, 'sentiment_agent'), (de+9*sa+0.01, 'blend'),
        ]
    elif mode == 'cached':
        # Data from cache: completes in ~5s instead of ~45s
        # Agents take normal time (~15s wall)
        de = 5.0
        return [
            (0.1, 'benchmark'), (0.5, 'price_history'), (de, 'fundamentals'),
            (de+10, 'risk_agent'), (de+11, 'value_agent'),
            (de+12, 'growth_momentum_agent'), (de+13, 'macro_regime_agent'),
            (de+15, 'sentiment_agent'), (de+15.01, 'blend'),
        ]
    else:
        s = 1.0
    de = 45.0 * s
    return [
        (0.4*s, 'benchmark'), (4.5*s, 'price_history'), (de, 'fundamentals'),
        (de+10*s, 'risk_agent'), (de+11*s, 'value_agent'),
        (de+12*s, 'growth_momentum_agent'), (de+13*s, 'macro_regime_agent'),
        (de+15*s, 'sentiment_agent'), (de+15*s+0.01, 'blend'),
    ]


def simulate(mode='normal'):
    events = build_scenario(mode)
    actual_total = events[-1][0]

    est_data_wall = _lp['data_gather']
    est_agents_wall = max(_lp['value_agent'], _lp['growth_momentum_agent'],
                          _lp['macro_regime_agent'], _lp['risk_agent'],
                          _lp['sentiment_agent'])
    est_blend = _lp['blend']

    data_keys = {'benchmark', 'price_history', 'fundamentals'}
    agent_keys = {'value_agent', 'growth_momentum_agent',
                  'macro_regime_agent', 'risk_agent', 'sentiment_agent'}

    step_done_at = {}
    n_steps_total = 9

    avg_total = _lp['avg_total']
    countdown = avg_total
    rate = 1.0

    def estimate_remaining(elapsed):
        """Phase-aware estimate of real seconds remaining.

        Data speed and agent speed are treated independently —
        cached data does NOT mean faster agents (different APIs).
        """
        d_done = {k: v for k, v in step_done_at.items() if k in data_keys}
        a_done = {k: v for k, v in step_done_at.items() if k in agent_keys}

        # ---- DATA PHASE (fundamentals not done yet) ----
        if 'fundamentals' not in d_done:
            frac = elapsed / est_data_wall if est_data_wall > 0 else 1.0
            if frac < 0.85:
                data_rem = est_data_wall - elapsed
            elif frac < 1.0:
                hedge = 1.0 + (frac - 0.85) / 0.15 * 0.5
                data_rem = max(3.0, (est_data_wall - elapsed) * hedge)
            else:
                overshoot = elapsed - est_data_wall
                grace = est_data_wall * 0.3 * math.exp(-overshoot / (est_data_wall * 0.4))
                data_rem = max(3.0, grace)
            return data_rem + est_agents_wall + est_blend

        # ---- DATA COMPLETE ----
        data_actual = d_done['fundamentals']

        # ---- AGENT PHASE ----
        # Agent speed is independent of data speed (different APIs/services).
        # Do NOT scale agent estimate by data_ratio.
        agents_elapsed = max(0, elapsed - data_actual)

        if len(a_done) >= 5:
            return est_blend

        if len(a_done) > 0:
            n = len(a_done)
            frac = n / (n + 1.0)
            if frac > 0.01:
                observed_agent_total = agents_elapsed / frac
            else:
                observed_agent_total = est_agents_wall
            trust = n / 5.0
            blended_total = observed_agent_total * trust + est_agents_wall * (1 - trust)
            agent_rem = max(0.5, blended_total - agents_elapsed)
        else:
            agent_rem = max(1.0, est_agents_wall - agents_elapsed)

        return agent_rem + est_blend

    # --- Simulation loop ---
    dt = 0.05
    elapsed = 0.0
    event_idx = 0
    last_print_sec = -1
    max_err = 0.0
    sum_abs_err = 0.0
    n_samples = 0

    print(f"\n{'='*78}")
    print(f"  TIMER v5 — mode={mode}")
    print(f"  Actual total: {actual_total:.1f}s | Initial estimate: {avg_total:.1f}s")
    print(f"{'='*78}")
    print(f"{'sec':>4}  {'display':>8}  {'actual':>8}  {'error':>7}  {'rate':>5}  {'steps':>5}  {'event':<25}")
    print(f"{'-'*4}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*25}")

    done = False
    while not done:
        # Fire events
        event_msg = ""
        while event_idx < len(events) and events[event_idx][0] <= elapsed:
            ev_time, ev_key = events[event_idx]
            step_done_at[ev_key] = ev_time
            event_msg = ev_key
            event_idx += 1

        if event_idx >= len(events) and elapsed > actual_total + 0.5:
            done = True

        # ---- CONTINUOUS RATE ADJUSTMENT (every tick) ----
        est_rem = estimate_remaining(elapsed)

        if countdown > 0.5 and est_rem > 0.5:
            target_rate = countdown / est_rem
            target_rate = max(0.05, min(5.0, target_rate))
            # Fast convergence: ~0.5s to reach 90% of target
            alpha = min(0.6, 3.0 * dt)
            rate += (target_rate - rate) * alpha
        elif countdown <= 0.5:
            # Near zero — hold steady, don't jitter
            rate = 0.05
        else:
            # est_rem near zero but countdown still has value — drain fast
            rate = max(rate, 2.0)

        countdown -= rate * dt
        countdown = max(0.0, countdown)

        # Simulate milestone progress percentage (like app.py's mp)
        # 0-42%: data gathering, 42-98%: agents (each ~11.2%), 98-100%: blend
        n_data_done = sum(1 for k in step_done_at if k in data_keys)
        n_agents_done = sum(1 for k in step_done_at if k in agent_keys)
        if 'blend' in step_done_at:
            sim_pct = 100
        elif n_agents_done > 0:
            sim_pct = 42 + n_agents_done * 11.2
        elif n_data_done >= 3:
            sim_pct = 42
        else:
            sim_pct = min(42, n_data_done * 14)

        # Progress floor: never show "finishing up" before agents ~done
        if sim_pct < 85:
            countdown = max(countdown, 3.0)

        display_remaining = countdown

        # Print every second
        sec = int(elapsed)
        if sec > last_print_sec:
            last_print_sec = sec
            actual_rem = max(0, actual_total - elapsed)
            err = display_remaining - actual_rem
            abs_err = abs(err)
            max_err = max(max_err, abs_err)
            sum_abs_err += abs_err
            n_samples += 1
            print(f"{sec:>4}  {display_remaining:>7.1f}s  {actual_rem:>7.1f}s  {err:>+6.1f}s  {rate:>4.2f}  {len(step_done_at):>3}/{n_steps_total}  {event_msg:<25}")

        elapsed += dt

    avg_err = sum_abs_err / n_samples if n_samples else 0
    print(f"\n  Final error: {countdown:.1f}s remaining when actual was 0.0s")
    print(f"  Max |error|: {max_err:.1f}s  |  Avg |error|: {avg_err:.1f}s")
    print(f"{'='*78}\n")
    return {'final': countdown, 'max_err': max_err, 'avg_err': avg_err}


if __name__ == '__main__':
    modes = ['normal']
    if '--fast' in sys.argv: modes = ['fast']
    elif '--slow' in sys.argv: modes = ['slow']
    elif '--mixed' in sys.argv: modes = ['mixed']
    elif '--cached' in sys.argv: modes = ['cached']
    elif '--all' in sys.argv: modes = ['normal', 'fast', 'slow', 'mixed', 'cached']
    results = {}
    for m in modes:
        results[m] = simulate(m)
    if len(results) > 1:
        print("\n" + "="*50)
        print("  SUMMARY")
        print("="*50)
        for m, r in results.items():
            print(f"  {m:>8}: final={r['final']:+.1f}s  max_err={r['max_err']:.1f}s  avg_err={r['avg_err']:.1f}s")
        print("="*50)
