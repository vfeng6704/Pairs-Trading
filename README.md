## Executive Summary

This project explored a pairs trading strategy using equity ETFs, specifically EWJ (Japan) and FEZ (Europe). Despite a historically strong correlation, the strategy produced a -9% average return per trade, with a final capital of ~$47.69M from an initial ~$51M investment, and a Sharpe ratio of 1.7. The results suggest the need for improved risk management and the incorporation of macro-based trading signals.

### Analytical Approach

- **Trade Signal**: Utilized a Z-score trading signal, entering positions when the Z-score between the ETFs deviated significantly from the mean.
- **Position Sizing**: Determined by the Z-score's strength and the hedge ratio between ETFs.
- **Trade Execution**: Positions entered when the cumulative drawdown was within limits and the Z-score exceeded thresholds; exited when Z-score reverted.
- **Return Calculation**: Based on the price difference between entry and exit, multiplied by the number of shares.

### Scenario Analysis

- The optimal Z-score thresholds were found to be 1 and -1, but results varied significantly with different thresholds.
- Revised the entry/exit logic to potentially improve returns, showing higher Sharpe ratio and win rate but also increased risk.

### Next Steps

- **Enhancements**: Consider incorporating macroeconomic factors to refine the trading strategy, potentially adjusting the Z-score signal with the spread between JPN and EZ interest rates.
- **Further Investigation**: Explore different hedging methodologies and assess the impact of incorporating overnight lending rates using Vector Autoregression (VAR).
