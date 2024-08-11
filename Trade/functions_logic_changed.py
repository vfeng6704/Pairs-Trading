### Defining functions used in testing our trade

from __future__ import print_function
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader as pdr
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter


def plot_series(df, title='Spread over Time', forecast=None):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Spread'], label='Spread', color = 'darkblue')
    if forecast is not None:
        plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
    plt.title(title)
    plt.ylabel('Spread')
    plt.legend()
    plt.show()

# Adapted code from this source: https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter/
def draw_date_coloured_scatterplot(etfs, prices):
    """
    Create a scatterplot of the two ETF prices, which is
    coloured by the date of the price to indicate the
    changing relationship between the sets of prices
    """
    # Create a yellow-to-red colourmap where yellow indicates
    # early dates and red indicates later dates
    plen = len(prices)
    colour_map = plt.cm.get_cmap('YlOrRd')
    colours = np.linspace(0.1, 1, plen)

    # Create the scatterplot object
    scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]],
        s=30, c=colours, cmap=colour_map,
        edgecolor='k', alpha=0.8
    )

    # Add a colour bar for the date colouring and set the
    # corresponding axis tick labels to equal string-formatted dates
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen//9].index]
    )
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.show()


def calc_slope_intercept_kalman(etfs, prices):
    """
    Utilise the Kalman Filter from the pyKalman package
    to calculate the slope and intercept of the regressed
    ETF prices.
    """
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack(
        [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]
    ).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )

    state_means, state_covs = kf.filter(prices[etfs[1]].values)
    return state_means, state_covs


def draw_slope_intercept_changes(prices, state_means):
    """
    Plot the slope and intercept changes from the
    Kalman Filte calculated values.
    """
    pd.DataFrame(
        dict(
            slope=state_means[:, 0],
            intercept=state_means[:, 1]
        ), index=prices.index
    ).plot(subplots=True)
    plt.show()

def identify_trade_periods(data, UPPERBOUND, LOWERBOUND, MIN_CONSECUTIVE_DAYS):
    """
    Identifies periods where the z-score is outside the defined bounds for at least MIN_CONSECUTIVE_DAYS consecutive days.
    """
    conditions = ((data['Z-Score'] > UPPERBOUND) & (data['Z-Score'] > 0)) | \
                 ((data['Z-Score'] < LOWERBOUND) & (data['Z-Score'] < 0))
    data['trade_condition'] = conditions
    
    # Identify consecutive days where trade conditions are met
    data['condition_cumsum'] = (data['trade_condition'] != data['trade_condition'].shift(1)).cumsum()
    trade_groups = data.groupby('condition_cumsum').cumcount() + 1

    # Only consider the groups that meet the trade_condition (True)
    data['condition_count'] = np.where(data['trade_condition'], trade_groups, 0)

    # Flag rows where trade can be executed (i.e., condition is met for at least MIN_CONSECUTIVE_DAYS)
    data['execute_trade'] = data['condition_count'] >= MIN_CONSECUTIVE_DAYS

    # Initialize columns for entry Z-score and trade status
    data['entry_zscore'] = np.nan
    data['in_trade'] = False

    in_trade = False
    entry_zscore = None

    for i in range(len(data)):
        if data.loc[i, 'execute_trade'] and not in_trade:
            # Start a new trade
            in_trade = True
            entry_zscore = data.loc[i, 'Z-Score']
            data.loc[i, 'entry_zscore'] = entry_zscore
            data.loc[i, 'in_trade'] = True
        elif in_trade:
            data.loc[i, 'entry_zscore'] = entry_zscore
            data.loc[i, 'in_trade'] = True
            # Check for exit condition
            if (entry_zscore < 0 and data.loc[i, 'Z-Score'] > entry_zscore) or \
               (entry_zscore > 0 and data.loc[i, 'Z-Score'] < entry_zscore):
                in_trade = False
                data.loc[i, 'in_trade'] = False
                data.loc[i, 'execute_trade'] = False  # Exit trade condition
                data.loc[i, 'condition_count'] = 0  # Reset condition count

                # Recalculate condition_count and execute_trade for subsequent rows
                for j in range(i + 1, len(data)):
                    if data.loc[j, 'trade_condition']:
                        data.loc[j, 'condition_count'] = data.loc[j - 1, 'condition_count'] + 1
                    else:
                        data.loc[j, 'condition_count'] = 0

                    data.loc[j, 'execute_trade'] = data.loc[j, 'condition_count'] >= MIN_CONSECUTIVE_DAYS
            else:
                data.loc[i, 'in_trade'] = True
                data.loc[i, 'entry_zscore'] = entry_zscore

    return data

# Process data to identify trade periods
def calculate_hedge_ratio(data, initial_state):
    """
    Calculate the dynamic hedge ratio using the Kalman Filter.
    """
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack([data['EWJ Price'], np.ones(data['EWJ Price'].shape)]).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=initial_state,
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )

    state_means, _ = kf.filter(data['FEZ Price'].values)
    return state_means[:, 0]  # Slopes as hedge ratios

# Process data to calculate hedge ratios during trading periods
def process_trading_periods(data):
    data['hedge_ratio'] = np.nan
    trading_indices = data.index[data['execute_trade']]
    
    for start, end in zip(trading_indices[:-1], trading_indices[1:]):
        if not data.loc[start, 'execute_trade']:  # Check if it's the beginning of a new trading period
            continue
        # Select data for the current trading period
        trading_data = data.loc[start:end]
        previous_day = data.loc[data.index[data.index < start][-1]]
        
        # Initial state for the Kalman filter from the last non-trade day
        initial_state = [0, 0]  # Placeholder for actual initial state logic
        hedge_ratios = calculate_hedge_ratio(trading_data, initial_state)
        data.loc[start:end, 'hedge_ratio'] = hedge_ratios
    
    return data


def calculate_trade_pnl(data, LONG):
    """
    Calculate PnL based on entry to exit positions and their respective price differences.
    """
    # Calculate daily returns for each ETF
    data['FEZ_return'] = data['FEZ Price'].pct_change()
    data['EWJ_return'] = data['EWJ Price'].pct_change()

    # Initialize columns for PnL calculation
    data['PnL_per_trade'] = 0
    data['daily_PnL_per_trade'] = 0
    data['long_shares'] = 0
    data['short_shares'] = 0
    data['long_allocation'] = 0
    data['short_allocation'] = 0
    data['capital_layover'] = 0
    data['return'] = 0

    # Prepare the hedge_ratio from the previous day to be used in calculations
    data['prev_hedge_ratio'] = data['hedge_ratio'].shift(1)
    data['prev_hedge_ratio'].fillna(method='bfill', inplace=True)

    in_trade = False

    for i in range(len(data)):
        if data.loc[i, 'in_trade'] and not in_trade:
            # Start of a new trade
            in_trade = True
            entry_index = i
            entry_price_ewj = data.loc[i, 'EWJ Price']
            entry_price_fez = data.loc[i, 'FEZ Price']

            # Calculate long and short allocations
            long_allocation = LONG * data.loc[i, 'Z-Score']
            short_allocation = -long_allocation * data.loc[i, 'prev_hedge_ratio']

            # Store allocations in separate columns
            data.loc[i, 'long_allocation'] = long_allocation
            data.loc[i, 'short_allocation'] = short_allocation
            data.loc[i, 'capital_layover'] = long_allocation + short_allocation

            # Determine shares based on long_etf column
            if data.loc[i, 'long_etf'] == 'EWJ':
                data.loc[i, 'long_shares'] = long_allocation / entry_price_ewj
                data.loc[i, 'short_shares'] = short_allocation / entry_price_fez
            else:
                data.loc[i, 'long_shares'] = long_allocation / entry_price_fez
                data.loc[i, 'short_shares'] = short_allocation / entry_price_ewj

        elif not data.loc[i, 'in_trade'] and in_trade:
            # End of the trade
            exit_index = i
            exit_price_ewj = data.loc[i, 'EWJ Price']
            exit_price_fez = data.loc[i, 'FEZ Price']

            # Calculate PnL based on entry and exit prices
            if data.loc[entry_index, 'long_etf'] == 'EWJ':
                long_pnl = data.loc[entry_index, 'long_shares'] * (exit_price_ewj - entry_price_ewj)
                short_pnl = data.loc[entry_index, 'short_shares'] * (entry_price_fez - exit_price_fez)
            else:
                long_pnl = data.loc[entry_index, 'long_shares'] * (exit_price_fez - entry_price_fez)
                short_pnl = data.loc[entry_index, 'short_shares'] * (entry_price_ewj - exit_price_ewj)

            total_pnl = long_pnl + short_pnl
            days_in_trade = exit_index - entry_index
            data.loc[exit_index, 'PnL_per_trade'] = total_pnl
            data.loc[exit_index, 'return'] = total_pnl / data.loc[entry_index, 'capital_layover']
            data.loc[exit_index, 'daily_PnL_per_trade'] = total_pnl / days_in_trade if days_in_trade > 0 else 0

            # Reset trade flag
            in_trade = False

    # If still in trade at the end of data, close it
    if in_trade:
        exit_index = len(data) - 1
        exit_price_ewj = data.loc[exit_index, 'EWJ Price']
        exit_price_fez = data.loc[exit_index, 'FEZ Price']

        # Calculate PnL based on entry and exit prices
        if data.loc[entry_index, 'long_etf'] == 'EWJ':
            long_pnl = data.loc[entry_index, 'long_shares'] * (exit_price_ewj - entry_price_ewj)
            short_pnl = data.loc[entry_index, 'short_shares'] * (entry_price_fez - exit_price_fez)
        else:
            long_pnl = data.loc[entry_index, 'long_shares'] * (exit_price_fez - entry_price_fez)
            short_pnl = data.loc[entry_index, 'short_shares'] * (entry_price_ewj - exit_price_ewj)

        total_pnl = long_pnl + short_pnl
        days_in_trade = exit_index - entry_index
        data.loc[exit_index, 'PnL_per_trade'] = total_pnl
        data.loc[exit_index, 'return'] = total_pnl / data.loc[entry_index, 'capital_layover']
        data.loc[exit_index, 'daily_PnL_per_trade'] = total_pnl / days_in_trade if days_in_trade > 0 else 0

    return data

def create_trade_summary_df(data):
    """
    Create a summary DataFrame for each trade, including specified columns.
    """
    # Ensure there are no NaN values in PnL_per_trade and capital_layover
    data = data.dropna(subset=['PnL_per_trade', 'capital_layover'])

    trade_summary = []

    in_trade = False
    entry_index = None

    for i in range(len(data)):
        if data.loc[i, 'in_trade'] and not in_trade:
            # Start of a new trade
            in_trade = True
            entry_index = i
        elif not data.loc[i, 'in_trade'] and in_trade:
            # End of the trade
            trade_summary.append({
                'date': data.loc[i, 'date'],
                'long_etf': data.loc[entry_index, 'long_etf'],
                'hedge_ratio': data.loc[entry_index, 'hedge_ratio'],
                'long_allocation': data.loc[entry_index, 'long_allocation'],
                'short_allocation': data.loc[entry_index, 'short_allocation'],
                'capital_layover': data.loc[entry_index, 'capital_layover'],
                'entry_zscore': data.loc[entry_index, 'entry_zscore'],
                'PnL_per_trade': data.loc[i, 'PnL_per_trade'],
                'daily_PnL_per_trade': data.loc[i, 'daily_PnL_per_trade'],
                'return': data.loc[i, 'PnL_per_trade'] / data.loc[entry_index, 'capital_layover']
            })
            in_trade = False

    trade_summary_df = pd.DataFrame(trade_summary)
    return trade_summary_df


def calculate_risk_metrics(trade_summary_df):
    """
    Calculate various risk metrics based on PnL per trade and daily PnL per trade.

    Parameters:
    - trade_summary_df (DataFrame): DataFrame containing the trade summary data.

    Returns:
    - metrics (dict): Dictionary containing various performance and risk metrics.
    """
    metrics = {}

    # Ensure there are no NaN values in PnL_per_trade and capital_layover
    trade_summary_df = trade_summary_df.dropna(subset=['PnL_per_trade', 'capital_layover'])

    # Total Return
    total_return = trade_summary_df['PnL_per_trade'].sum()
    metrics['Total Return'] = total_return
    
    # Average Return per Trade
    average_return_per_trade = trade_summary_df['PnL_per_trade'].mean()
    metrics['Average Return per Trade'] = average_return_per_trade
    
    # Average Return Rate
    number_of_trades = trade_summary_df['PnL_per_trade'].count()
    avg_return_rate = trade_summary_df['PnL_per_trade'].sum() / trade_summary_df['capital_layover'].sum()
    metrics['Average Return Rate'] = avg_return_rate

    # Sharpe Ratio based on daily PnL
    daily_returns = trade_summary_df['daily_PnL_per_trade']
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    metrics['Sharpe Ratio'] = sharpe_ratio

    # Maximum Drawdown
    cumulative_pnl = trade_summary_df['PnL_per_trade'].cumsum()
    drawdown = cumulative_pnl - cumulative_pnl.cummax()
    max_drawdown = drawdown.min()
    metrics['Maximum Drawdown'] = max_drawdown

    # Win Rate
    number_of_win_trades = (trade_summary_df['PnL_per_trade'] > 0).sum()
    win_rate = number_of_win_trades / number_of_trades
    metrics['Win Rate'] = win_rate


    # Average Win and Loss
    average_win = trade_summary_df.loc[trade_summary_df['PnL_per_trade'] > 0, 'PnL_per_trade'].mean()
    average_loss = trade_summary_df.loc[trade_summary_df['PnL_per_trade'] < 0, 'PnL_per_trade'].mean()
    metrics['Average Win'] = average_win
    metrics['Average Loss'] = average_loss

    # Value at Risk (VaR)
    var_95 = np.percentile(daily_returns, 5)
    metrics['VaR (95%)'] = var_95

    # Conditional Value at Risk (CVaR)
    cvar_95 = daily_returns[daily_returns <= var_95].mean()
    metrics['CVaR (95%)'] = cvar_95

    return metrics


def macro_trading_periods(data, UPPERBOUND, LOWERBOUND, MIN_CONSECUTIVE_DAYS, DRAWDOWN_THRESH):
    """
    Identifies periods where the z-score is outside the defined bounds for at least MIN_CONSECUTIVE_DAYS consecutive days
    and where drawdowns did not fall below DRAWDOWN_THRESH in the previous MIN_CONSECUTIVE_DAYS rows.
    """
    conditions = ((data['Z-Score'] > UPPERBOUND) & (data['Z-Score'] > 0)) | \
                 ((data['Z-Score'] < LOWERBOUND) & (data['Z-Score'] < 0))
    data['trade_condition'] = conditions
    
    # Identify consecutive days where trade conditions are met
    data['condition_cumsum'] = (data['trade_condition'] != data['trade_condition'].shift(1)).cumsum()
    trade_groups = data.groupby('condition_cumsum').cumcount() + 1

    # Only consider the groups that meet the trade_condition (True)
    data['condition_count'] = np.where(data['trade_condition'], trade_groups, 0)

    # Check if drawdowns are above the threshold for the last MIN_CONSECUTIVE_DAYS
    def drawdown_condition(index):
        if index < MIN_CONSECUTIVE_DAYS:
            return False
        fez_drawdown_ok = all(data['FEZ_drawdown'].iloc[index - MIN_CONSECUTIVE_DAYS:index] <= DRAWDOWN_THRESH)
        ewj_drawdown_ok = all(data['EWJ_drawdown'].iloc[index - MIN_CONSECUTIVE_DAYS:index] <= DRAWDOWN_THRESH)
        return fez_drawdown_ok and ewj_drawdown_ok

    # Ensure the index is an integer and keep the original index as a column
    data = data.reset_index(drop=False)

    data['execute_trade'] = data.apply(
        lambda row: row['condition_count'] >= MIN_CONSECUTIVE_DAYS and drawdown_condition(row.name),
        axis=1
    )

    # Initialize columns for entry Z-score and trade status
    data['entry_zscore'] = np.nan
    data['in_trade'] = False

    in_trade = False
    entry_zscore = None

    for i in range(len(data)):
        if data.loc[i, 'execute_trade'] and not in_trade:
            # Start a new trade
            in_trade = True
            entry_zscore = data.loc[i, 'Z-Score']
            data.loc[i, 'entry_zscore'] = entry_zscore
            data.loc[i, 'in_trade'] = True
        elif in_trade:
            data.loc[i, 'entry_zscore'] = entry_zscore
            data.loc[i, 'in_trade'] = True
            # Check for exit condition
            if (entry_zscore < 0 and data.loc[i, 'Z-Score'] > entry_zscore) or \
               (entry_zscore > 0 and data.loc[i, 'Z-Score'] < entry_zscore):
                in_trade = False
                data.loc[i, 'in_trade'] = False
                data.loc[i, 'execute_trade'] = False  # Exit trade condition
                data.loc[i, 'condition_count'] = 0  # Reset condition count

                # Recalculate condition_count and execute_trade for subsequent rows
                for j in range(i + 1, len(data)):
                    if data.loc[j, 'trade_condition']:
                        data.loc[j, 'condition_count'] = data.loc[j - 1, 'condition_count'] + 1
                    else:
                        data.loc[j, 'condition_count'] = 0

                    data.loc[j, 'execute_trade'] = data.loc[j, 'condition_count'] >= MIN_CONSECUTIVE_DAYS and drawdown_condition(j)
            else:
                data.loc[i, 'in_trade'] = True
                data.loc[i, 'entry_zscore'] = entry_zscore

    return data


def calculate_drawdown(prices):
    # Calculate cumulative maximum
    cummax = prices.cummax()
    # Calculate drawdown
    drawdown = (prices / cummax - 1) * 100  # Convert drawdown to percentage
    return drawdown

