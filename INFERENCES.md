# Inferences from the trained Crossy Road agents

Snapshot of the six trained checkpoints in this repository. Everything below is
derived from the artefacts that are actually committed:

- `results/<algo>.json` — training curves, hyperparameters, post-training evaluation
- `results/evaluations.npz` — PPO's Stable-Baselines3 `EvalCallback` history
- `results/train_<algo>.log` — raw training logs
- `evaluate.py` output in `results/eval_all.log`

The shared budget is **100,000 environment steps** (not 20k — the line in
`notes.md` saying 20k is stale), seed 0, greedy evaluation, 20 episodes, and the
same deterministic `GridEnv`. Best-eval checkpointing is enabled for the five
from-scratch algorithms; PPO restores `best_model.zip`.

---

## 1. Final table (after fixing the ranking)

Ranks are unique now. The key is: converged → success rate → mean reward →
shorter path → first convergence → timestep of the checkpoint that was actually
restored. Wall-clock time is displayed but deliberately **not** used for
ranking.

| # | Algorithm | Category | Eval mean | Success | Mean length | First converged | Restored checkpoint | Train time |
|---|-----------|----------|-----------|---------|-------------|-----------------|---------------------|------------|
| 1 | PPO | policy-based | **134.20** | 100% | **8.0** | 15,000 | 15,000 | 55.2 s |
| 2 | A3C | hybrid | **134.20** | 100% | **8.0** | 15,000 | **35,000** | 55.4 s |
| 3 | A2C | hybrid | **134.20** | 100% | **8.0** | 50,000 | 50,000 | 103.7 s |
| 4 | DQN | value-based | 133.80 | 100% | 12.0 | 30,000 | 30,000 | 271.5 s |
| 5 | Double DQN | value-based | 133.80 | 100% | 12.0 | 90,000 | 90,000 | 191.8 s |
| 6 | REINFORCE | policy-based | 133.50 | 100% | 15.0 | 35,006 | 65,006 | 72.1 s |

### The ranking bug that was fixed

The previous implementation used **dense ranking**, so identical sort keys
shared a rank. PPO and A3C genuinely both first met the convergence criterion at
15,000 steps, so both rendered as `#1`.

That was misleading for a second reason: A3C's 15,000-step convergence was
transient (mean 134.0). Training later collapsed, and the checkpoint that was
actually saved came from **35,000 steps** (mean 134.2). PPO's restored
checkpoint really was the 15,000-step one. The ranking now uses the restored
checkpoint timestep to break the first-convergence tie, and ranks are strictly
`#1..#N`. In this run that gives **PPO #1, A3C #2**.

---

## 2. The most important inference: reward is only path length

For a goal episode, every upward move gives `+5`, every step costs `-0.1`, and
the goal gives `+100`. Reaching row 0 from row 7 always needs exactly 7 upward
moves, so for any successful episode:

```
reward = 100 + 5*7 - 0.1*steps = 135 - 0.1*steps
```

Check this against every converged model:

| Path length | Predicted reward | Observed |
|-------------|------------------|----------|
| 8 steps | 135 - 0.8 = **134.2** | PPO, A3C, A2C |
| 12 steps | 135 - 1.2 = **133.8** | DQN, Double DQN |
| 15 steps | 135 - 1.5 = **133.5** | REINFORCE |

So the reward differences are **not** differences in goal-reaching ability. All
six converged policies reach the goal every time; they differ only in how many
wasted lateral/stay steps they take. The whole 134.2 → 133.5 spread is 0.7
reward, i.e. **7 extra non-upward steps**.

This also explains why the top three are effectively tied: they all found the
same 8-step timing pattern (7 UP + 1 wait/dodge). The ranking between them is
therefore mostly about *when* they found that pattern and whether the saved
checkpoint still had it.

---

## 3. Per-model notes

### PPO — #1, converged 15k, best 15k

**Curve:** `-5.4 @ 5k → 9.5 @ 10k → 134.2 @ 15k → 134.2 @ 20k → 9.4 @ 25k →
134.2 @ 30k → 134.2 for the rest`.

**What happened:** PPO found the optimal 8-step path at 15k, briefly fell back to
a 10-step non-goal policy at 25k, then held the optimum from 30k to 100k.

**Why it is best:**

- Clipped surrogate objective + GAE + 10 epochs per 2,048-step rollout gives
  several stable passes over each batch of experience.
- `ent_coef=0.05` keeps enough exploration to recover from the 25k dip.
- SB3's `EvalCallback` saved the first 134.2 checkpoint, so the deployed policy
  is the one from 15k, not the final weights.
- It reached the best *and* earliest restored checkpoint, which is why it wins
  the tie-break.

**Caveat:** PPO's training reward history is only SB3's rolling last-100 episode
buffer, not a full episode log, so its training summary is not comparable to the
others.

---

### A3C — #2, converged 15k, best 35k

**Curve:** `-5.1 @ 5k → -5.1 @ 10k → 134.0 @ 15k → 9.4 @ 20k → 14.1 @ 25k →
-5.1 @ 30k → 134.2 @ 35k → 134.2 @ 40k → -5.1 @ 45k..100k`.

**What happened:** A3C touched the goal policy at 15k, lost it, recovered at
35–40k, then collapsed permanently to a 2-step crash (`-5.1`, length 2). The
saved model is the 35k snapshot.

The `-5.1 / length 2` signature is informative: it is consistent with
`UP → UP → collision` (one +4.9 progress step, then `-10` crash). The workers
converged onto a deterministic local optimum rather than a goal policy.

**Why it converged early:**

- Four workers explore different trajectories, so the shared model sees more
  decorrelated experience than A2C's single stream. That is the most likely
  reason it beat A2C to the goal by 35k steps.
- With four workers, the wall-clock is also competitive (55.4s, about the same
  as PPO), despite the Python-level env.

**Why it then collapsed:**

- `max_grad_norm=40.0` is far looser than A2C's 0.5. Large updates are allowed
  when the advantage estimate is noisy.
- Workers update the same global parameters under a lock, so a bad update from
  one worker is immediately visible to the others. There is no replay buffer to
  retain the good 15k/35k behaviour.
- Entropy `0.01` is weak once the policy becomes nearly deterministic, so it
  cannot easily escape the crash local optimum.
- Best-checkpoint restore is the only reason the delivered model is good at all;
  the final live policy would score 0% success.

This is the clearest example of why "converged at step X" is not the same as
"stayed converged".

---

### A2C — #3, converged 50k, best 50k

**Curve:** `-5.1 for 5k–45k → 134.2 @ 50k → -5.1 @ 55–60k → 9.5 @ 65k →
9.2 @ 70k → -1.0 @ 75k → 9.4 @ 80k → 14.1 @ 85–90k → 9.3 @ 95k → 9.5 @ 100k`.

**What happened:** A2C was stuck at a 2-step crash for 45k steps, found the goal
once at 50k, then lost it and never recovered. The saved model is that single
50k snapshot.

**Why it converged later than A3C:** it has the same actor-critic shape and
learning rate but only one environment stream. With 5-step rollouts, updates are
highly correlated and exploration is weaker than A3C's four worker streams.
That is enough to delay first success by ~35k steps.

**Why it still collapsed:** the same on-policy failure mode as A3C. Low entropy,
no replay, and a deterministic task let the policy fall into the
"UP, UP, crash" pattern. Its tighter gradient clip (0.5) reduced update size but
did not prevent the policy from becoming deterministic and wrong.

A2C is therefore a good illustration that **stability is not just a clipping
constant**: a single correlated stream can still lock into the wrong local
optimum.

---

### DQN — #4, converged 30k, best 30k

**Curve:** `-0.9 @ 5k → -5/-10 @ 10–20k (100-step timeouts) → 8.9 @ 25k →
133.8 @ 30k → -0.4 @ 35k → 133.8 @ 40k → 14 @ 45–55k → -5/-10 @ 60–75k →
133.3 @ 80–100k`.

**What happened:** DQN found a 12-step goal policy at 30k, forgot it twice,
then recovered a slightly worse 17-step policy from 80k onward. Best snapshot:
30k, 12 steps.

**Why it found the goal relatively early (30k):**

- Soft target updates every step (`tau=0.01`) plus **two** gradient steps per
  environment step and batch size 128 propagate value information quickly.
- Epsilon at 30k is still around 0.6, so the successful trajectory is
  discovered by exploration, not by the final greedy schedule.
- The replay buffer lets it reuse the first successful transitions, which is why
  it can recover at 80k after bad stretches.

**Why it is not higher:**

- The retained policy is 12 steps, not the 8-step optimum, so it loses on reward
  and path length.
- The curve is unstable: it oscillates between goal policies and timeout/crash
  policies. The 100-step `-10` evaluations are episodes where the greedy policy
  never moves upward.
- It is the slowest in wall-clock (271.5s) because of 2 gradient steps × batch
  128 plus a parameter-wise soft update every environment step.

DQN's high training success rate (23.5%) is therefore not evidence of a better
policy; it reflects a long training run with many short crash episodes plus many
goal episodes, measured on the *behaviour* policy (epsilon-greedy), not the
greedy evaluation policy.

---

### Double DQN — #5, converged 90k, best 90k

**Curve:** stuck at `-5.1` through 35k, first real progress at 40k (`4.2`), then
timeouts and crashes until 85k, then **133.8 from 90k to 100k**.

**What happened:** Double DQN took three times longer than DQN to first reach
the goal, but once it did, it stayed there for the last 10k steps and produced a
stable 12-step policy.

**Why it converged later:**

- Hard target sync every 500 steps plus a separate online/target split slows
  value propagation compared with DQN's soft update every step.
- Its epsilon schedule reaches 0.05 only at 60k (`epsilon_decay=0.6`), so the
  greedy policy is not fully exploited until late.
- The double estimator reduces overestimation, which is good for stability but
  can make Q-values more conservative/slower to separate the goal action from
  the crash actions on this small reward scale.

**Why it is still below DQN:** same 12-step path (133.8), but it first converged
at 90k, so it loses the sample-efficiency tie-break.

**Important confound:** DQN and Double DQN do **not** differ only in the
bootstrap target in this code. `VALUE_HPARAMS` gives them different learning
rates (5e-4 vs 1e-3), batch sizes (128 vs 64), target update rules (soft τ=0.01
every step vs hard sync every 500), epsilon decay (0.7 vs 0.6), and gradient
steps (2 vs 1). So the DQN vs DDQN difference cannot be attributed cleanly to
Double Q-learning alone. A controlled comparison would keep all of those equal
and change only the target.

---

### REINFORCE — #6, converged 35k, best 65k

**Curve:** `-5.1 @ 5k → 4.4 @ 10–15k → 13.4 @ 20–30k → 133.1 @ 35–40k →
13.4 @ 45k → 133.1 @ 50–60k → 133.5 @ 65k → 133.1/133.5 oscillating to 100k`.

**What happened:** REINFORCE reached the goal at 35k but oscillated between goal
and near-miss policies until 65k, then held a 15-step policy. It never found the
8- or 12-step path.

**Why it converged reasonably early:** each episode produces a full Monte Carlo
return and one policy-gradient update, so the dense progress reward gives a
direct learning signal. In this tiny deterministic environment, that is enough
to find *a* goal policy after ~35k steps.

**Why it is last:**

- No critic/baseline: returns have high variance, so updates are noisy and the
  policy moves between goal and non-goal behaviours.
- The per-episode return normalisation helps, but cannot remove the variance of
  a 15-step credit assignment problem.
- It settles on a 15-step policy (8 wasted steps), which is the worst path
  efficiency of the converged models.
- Best checkpoint comes from 65k, not the first 35k convergence, because the
  later policy had a higher mean reward (133.5 vs 133.1).

REINFORCE is the best example of "sample-efficient enough to find the goal, but
not precise enough to optimise the path".

---

## 4. Why some models converged earlier and some later

### Family effect

1. **On-policy policy gradient (PPO, A3C, A2C)** directly changes the action
   probabilities. The dense `+5` progress reward gives a clear gradient, so
   these methods can lock onto a successful action sequence quickly. All three
   reached the goal between 15k and 50k.
2. **Off-policy value-based (DQN, DDQN)** have to learn a Q-function and wait
   for exploration to expose the goal. They reached it at 30k and 90k. Once
   found, replay lets them retain and revisit successful transitions, which is
   why their late curves are flatter than the on-policy ones.
3. **Monte Carlo policy gradient (REINFORCE)** found the goal at 35k, between
   the two, but with the noisiest curve and the worst final path.

### Data diversity effect

A3C (4 workers) found the goal at 15k; A2C (1 stream) took 50k. The architecture,
learning rate, rollout length, value coefficient and entropy coefficient are the
same. The main difference is that A3C's four workers generate more independent
trajectories and therefore a less correlated gradient. That is strong evidence
that **exploration diversity, not the actor-critic loss alone, drove the early
A3C result**.

PPO also found it at 15k, but through a different mechanism: 2,048-step
rollouts, 10 epochs of clipped updates, and a larger entropy bonus (0.05).

### Exploration schedule effect

DQN/DDQN are the only methods whose behaviour policy is explicitly
epsilon-greedy. Their "convergence time" is therefore partly a schedule
artefact:

- DQN found the goal at 30k while epsilon was still high (~0.6), so the finding
  was exploration-driven.
- DDQN did not stabilise until 90k, well after epsilon reached 0.05 at 60k. So
  epsilon decay alone does **not** explain the delay; target-network lag and the
  double estimator are the more likely causes.

### Best-checkpoint effect (the big one)

"First converged" and "best checkpoint" are different numbers for three models:

| Algorithm | First converged | Restored checkpoint | Why they differ |
|-----------|-----------------|---------------------|-----------------|
| A3C | 15,000 | 35,000 | 15k was transient (134.0); 35k was better (134.2) and later collapsed |
| REINFORCE | 35,006 | 65,006 | 35k policy was 19 steps (133.1); 65k was 15 steps (133.5) |
| PPO | 15,000 | 15,000 | first best was already optimal |
| A2C | 50,000 | 50,000 | only one good snapshot |
| DQN | 30,000 | 30,000 | later snapshots were worse |
| DDQN | 90,000 | 90,000 | converged late and held |

Without best-checkpoint restore, A3C, A2C and DQN would all be reported as
**not converged** at 100k. This is a crucial inference: the reported final scores
measure the **best policy seen**, not the **final policy learned**. The training
curves are therefore essential; the eval table alone hides catastrophic
forgetting.

---

## 5. Why the top three found 8 steps, DQN/DDQN 12, REINFORCE 15

A plausible interpretation (consistent with the curves, not proven by a single
run):

- **8-step policies** are the shortest safe timing pattern on this fixed track:
  7 UP moves plus one wait/dodge. On-policy actor-critic methods optimise the
  stochastic policy directly and can discover this precise timing; PPO's
  entropy bonus and A3C's worker diversity help. A2C found the same pattern once,
  then lost it.
- **12-step policies** add four conservative wait steps. Q-learning's
  bootstrapped values favour a safer, easier-to-learn timing pattern: extra
  waits reduce the chance that a slight timing error becomes a collision, and
  are easier to propagate through a replay buffer than a precise 8-step
  sequence.
- **15-step REINFORCE** adds even more waiting because the high-variance
  policy gradient cannot reliably assign credit to the exact step where a
  dodge is needed. A longer, more forgiving pattern is easier to reinforce.

This is the main qualitative story: **all converged agents learned to survive;
the better-ranked ones learned to waste fewer steps.**

---

## 6. Wall-clock notes

| Algorithm | Time | Main cost driver |
|-----------|------|------------------|
| PPO | 55.2 s | 49 SB3 iterations, but vectorised/optimised implementation |
| A3C | 55.4 s | 4 worker processes; parallel env stepping, lock contention |
| REINFORCE | 72.1 s | 8,626 episodes, one update per episode |
| A2C | 103.7 s | one update per 5-step rollout, single process |
| Double DQN | 191.8 s | 1 gradient step/step, batch 64, hard target sync |
| DQN | 271.5 s | 2 gradient steps/step, batch 128, soft update every step |

DQN is ~1.4× slower than Double DQN almost entirely because it does twice the
gradient work per environment step with a larger batch, plus a per-step soft
update. A3C's wall-clock is close to PPO despite four Python processes because
the grid step and the tiny MLP are cheap; process coordination and the update
lock eat most of the parallel gain.

Wall-clock is **not** a clean algorithmic comparison: process count, BLAS
threading, lock contention and machine load all matter. The `train_duration_sec`
field is best treated as a rough cost indicator.

---

## 7. Caveats — what these results do **not** show

1. **The evaluation is one trajectory repeated 20 times.** `GridEnv.reset()`
   always starts at `(3, 7)` with the same car positions and ignores the seed.
   Every evaluation episode is identical, so `std_reward ≈ 2.8e-14` is float
   noise, not policy variance, and `success_rate = 100%` means "this one
   deterministic episode succeeded 20 times".
2. **No generalisation claim is possible.** There is no randomised start, no
   randomised car layout, and no held-out track. A model can memorise the
   timing for this exact track.
3. **Different algorithms use different hyperparameters.** This is defensible
   because the parameters mean different things, but it means "algorithm X is
   better" is really "algorithm X with this configuration is better".
4. **DQN vs Double DQN is confounded** (see above): several hyperparameters
   differ, not just the target.
5. **The reported scores are best-checkpoint scores.** The final live policies
   for A3C, A2C and DQN are much worse. A separate "final weights" column would
   make the forgetting explicit.
6. **Training success rates are not comparable.** They are measured on each
   algorithm's behaviour policy (epsilon-greedy for DQN/DDQN, stochastic for the
   policy-gradient methods), with different entropy and exploration settings.
   They should not be used to rank final policy quality.
7. **One seed only.** Seed 0 for everything. RL results on a single seed are
   indicative, not conclusive.
8. **PPO uses 100,352 steps** (SB3 rounds to whole rollouts) while the others
   use ~100,000; the difference is negligible but not zero.
9. **PPO's `results/ppo.json` training summary** is only the last 100 episodes
   from SB3's internal buffer, not a full training log.
10. **`notes.md` has two stale claims:** the budget is 100k, not 20k, and PPO
    *does* have an in-training evaluation hook (`EvalCallback` in `train.py`).
    The "no in-training evaluation hook" note in `results/ppo.json` is also
    outdated.

---

## 8. Suggested next experiments

1. **Randomise the reset** (agent column and/or car offsets) so evaluation is a
   distribution, not one trajectory. This is the single highest-value change.
2. **Report final weights separately from best checkpoint** so catastrophic
   forgetting is visible in the table.
3. **Run 5+ seeds per algorithm** and report mean ± std across seeds.
4. **Control DQN vs DDQN** by using identical lr, batch, epsilon schedule and
   target-update cadence; change only the bootstrap target.
5. **Add an entropy/exploration ablation** for A2C/A3C to test whether the
   collapse is driven by `entropy_coef=0.01` and A3C's `max_grad_norm=40`.
6. **Log the actual action sequence** of the best policy. The 8/12/15-step
   inference could then be verified exactly (how many UP vs LEFT/RIGHT/STAY).
7. **Use a fixed evaluation track plus a random track** to separate timing
   memorisation from a general dodge policy.

---

## 9. One-paragraph summary

All six converged models reach the goal on the deterministic evaluation track;
their score differences are entirely path efficiency (134.2 for 8 steps, 133.8
for 12, 133.5 for 15). PPO is the best overall because it found the optimal
8-step path at 15k and its restored checkpoint is still that policy. A3C matched
it but only transiently: it first converged at 15k, collapsed, recovered at 35k,
then collapsed again, so its saved model is a 35k snapshot. A2C found the same
optimal path once at 50k and lost it. DQN found a 12-step policy early (30k) but
oscillated; Double DQN took until 90k but was stable afterwards. REINFORCE found
a goal policy at 35k but never refined it, settling on 15 steps. The deeper
lesson is that the final evaluation table mostly measures **which algorithm
produced the best checkpoint**, while the training curves show that on-policy
methods found good policies early but forgot them, and off-policy methods took
longer to find them but retained them better.
