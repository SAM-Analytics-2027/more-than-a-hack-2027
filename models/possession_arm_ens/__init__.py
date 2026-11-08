"""possession_arm_ens -- posterior-averaging ensemble closer (family 7).

The ensemble is itself under the IDENTICAL protocol: for every nested-LOO fold
it averages the held-out posteriors of the ingredient families (each fitted
without fold i and with early stop on fold i+1 -- the very same protocol), so
the composed posterior for a held-out file has seen no information beyond what
the ingredients legitimately saw. The averaged posterior then runs through the
SHARED Viterbi decoder whose parameters come from the shared external-only
coordinate search on the ENSEMBLE's own composed posteriors; competition-time
posteriors are the equal-weight average of every ingredient family's FINAL
artifact on the target match.

No new fitting happens here: composition + decode + search + predict.
"""
