# A Rawlsian Maximin Reward Prototype for Fair Driving Norms in HighwayEnv

## Minimal research question

**Can a Rawlsian maximin reward improve the minimum safety-mobility experience of vehicles in a merge scenario?**

## Theoretical lineage

- **RAWL·E**: Rawlsian ethics module and reward shaping (implemented here as a Gymnasium wrapper).
- **SYMPLEX**: future human-in-the-loop symbolic norm learning.
- **Agent-directed Runtime Norm Synthesis**: future runtime norm revision.

## Scope of this prototype

First version implements **RAWL·E-style reward shaping** only: no training, no human feedback, no symbolic norms.

## Vehicle experience (v1)

For each vehicle \(i\):

\[
\text{experience}_i = \text{normalized\_speed}_i - \text{collision\_penalty}_i
\]

- `normalized_speed_i = speed_i / speed_normalizer`
- `collision_penalty_i = 1.0` if crashed, else `0.0`

## Rawlsian shaping signal (v1)

Let \(m_t = \min_i \text{experience}_i\) at step \(t\), and \(m_{t-1}\) the previous minimum.

- If \(m_t > m_{t-1}\): `rawlsian_reward = +xi`
- If \(m_t < m_{t-1}\): `rawlsian_reward = -xi`
- If equal: `rawlsian_reward = 0`

Total reward: `original_reward + rawlsian_reward`.

## Extensions (roadmap)

- SVO reward
- Human feedback
- Symbolic norm representation
- Environments: `merge-v0` → `roundabout-v0`
- Trained RL agents instead of random policy
