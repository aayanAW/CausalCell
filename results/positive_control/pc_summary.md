# Positive-control scorecard

**Branch: A**  (8 pass / 0 inversion / 8 gates)  git `24e7b39`

Decisive block = CROSS-SPLIT (GT on held-out cells, ACE on disjoint cells).

| stratum | N | panel | GTedges | real F1 (xsplit) [CI] | null p97.5 | overlay | corr_max | in-sample real | C1 C2 C3 C4 | FDRq | PASS |
|--|--|--|--|--|--|--|--|--|--|--|--|
| S0 | 200 | 200 | 1920 | 0.4329 [0.4302,0.4355] | 0.1731 | 0.4223 | 0.1273 | 0.4403 | 1 1 1 1 | 0.0 | YES |
| S1 | 200 | 199 | 2057 | 0.3689 [0.3631,0.3746] | 0.1042 | 0.3499 | 0.1736 | 0.3860 | 1 1 1 1 | 0.0 | YES |
| S2 | 200 | 200 | 2647 | 0.4061 [0.4011,0.4112] | 0.1484 | 0.3889 | 0.1103 | 0.4111 | 1 1 1 1 | 0.0 | YES |
| S3 | 200 | 9 | — | SKIP: panel<20 | | | | | | | |
| S4 | 200 | 117 | 999 | 0.3546 [0.3478,0.3614] | 0.1236 | 0.3380 | 0.1663 | 0.3615 | 1 1 1 1 | 0.0 | YES |
| S0 | 500 | 500 | 6908 | 0.3797 [0.3767,0.3827] | 0.1113 | 0.3649 | 0.0916 | 0.3916 | 1 1 1 1 | 0.0 | YES |
| S1 | 500 | 199 | 2057 | 0.3689 [0.3631,0.3746] | 0.1042 | 0.3499 | 0.1736 | 0.3860 | 1 1 1 1 | 0.0 | YES |
| S2 | 500 | 385 | 5157 | 0.4017 [0.3975,0.4059] | 0.1302 | 0.3852 | 0.0945 | 0.4108 | 1 1 1 1 | 0.0 | YES |
| S3 | 500 | 9 | — | SKIP: panel<20 | | | | | | | |
| S4 | 500 | 117 | 999 | 0.3546 [0.3478,0.3614] | 0.1236 | 0.3380 | 0.1663 | 0.3615 | 1 1 1 1 | 0.0 | YES |

cond1 real CI-lo>null p97.5 · cond2 margin>MDE(0.010) · cond3 real>overlay (paired p<0.05) · cond4 real>=max(corr_ctrl,corr_all).
Branch A requires a cross-split gate to pass all four + FDR q<=0.05 and not be low-power.