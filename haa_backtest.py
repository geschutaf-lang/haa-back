"""
HAA (Hybrid Asset Allocation) 변형 백테스트
- 원래 HAA: 카나리아(TIP) 신호 → 공격(SPY) / 방어(SHY/IEF/BIL)
- 변형 HAA: 공격 시 SPY 대신 S&P500 종목 중 12개월 평균 모멘텀 Top1 선택

설치: pip install yfinance pandas numpy matplotlib
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
START = '2007-01-01'
END   = '2024-12-31'
TOP_N = 1  # 공격 포트폴리오에서 선택할 종목 수 (1~5 조절 가능)

# S&P500 대표 종목 (섹터별 분산)
SP500_UNIVERSE = [
    # IT
    'AAPL','MSFT','NVDA','GOOGL','META','AVGO','ORCL','ADBE','CRM','AMD',
    # 금융
    'JPM','V','MA','BAC','WFC','GS','MS','BLK','AXP','BRK-B',
    # 헬스케어
    'LLY','UNH','JNJ','ABBV','MRK','TMO','ABT','DHR','BMY','AMGN',
    # 소비재
    'AMZN','TSLA','HD','MCD','NKE','SBUX','TGT','LOW','TJX','BKNG',
    # 에너지
    'XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY',
    # 산업재
    'CAT','HON','UPS','BA','GE','MMM','LMT','RTX','DE','EMR',
    # 필수소비재
    'WMT','PG','KO','PEP','COST','CL','GIS','MO','PM',
    # 유틸리티/리츠
    'NEE','DUK','SO','AMT','PLD','CCI','EQIX','SPG','PSA','O',
    # 통신
    'T','VZ','TMUS','NFLX','DIS','CMCSA',
]

DEFENSIVE = ['SHY', 'IEF', 'BIL']  # 방어 자산 (모멘텀 상위 1개 선택)
CANARY    = 'TIP'                   # 카나리아 자산
BENCHMARK = 'SPY'                   # 비교용 벤치마크


# ──────────────────────────────────────────────
# 1. 데이터 다운로드
# ──────────────────────────────────────────────
def download_data():
    tickers = list(set(SP500_UNIVERSE + DEFENSIVE + [CANARY, BENCHMARK]))
    print(f"데이터 다운로드 중... ({len(tickers)}개 티커)")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=True)['Close']
    monthly = raw.resample('ME').last()
    # 절반 이상 데이터 없는 종목 제거
    monthly = monthly.dropna(axis=1, thresh=len(monthly) // 2)
    print(f"유효 종목 수: {monthly.shape[1]}개")
    print(f"기간: {monthly.index[0].date()} ~ {monthly.index[-1].date()}")
    return monthly


# ──────────────────────────────────────────────
# 2. Keller식 평균 모멘텀 계산
#    = (1개월 수익 + 3개월 수익 + 6개월 수익 + 12개월 수익) / 4
# ──────────────────────────────────────────────
def calc_momentum(prices: pd.Series) -> pd.Series:
    p = prices.dropna()
    mom = pd.Series(np.nan, index=p.index)
    for i in range(12, len(p)):
        r1  = p.iloc[i] / p.iloc[i-1]  - 1
        r3  = p.iloc[i] / p.iloc[i-3]  - 1
        r6  = p.iloc[i] / p.iloc[i-6]  - 1
        r12 = p.iloc[i] / p.iloc[i-12] - 1
        mom.iloc[i] = (r1 + r3 + r6 + r12) / 4
    return mom


# ──────────────────────────────────────────────
# 3. 백테스트 엔진
# ──────────────────────────────────────────────
def run_backtest(monthly: pd.DataFrame, use_stocks: bool = True, top_n: int = TOP_N):
    """
    use_stocks=True  → HAA 변형: S&P500 개별 종목 모멘텀 Top N
    use_stocks=False → 원래 HAA:  공격 자산 = SPY
    """
    # 카나리아 모멘텀 사전 계산
    if CANARY not in monthly.columns:
        raise ValueError(f"{CANARY} 데이터 없음")
    canary_mom = calc_momentum(monthly[CANARY])

    # 종목별 모멘텀 사전 계산 (속도 최적화)
    print(f"모멘텀 계산 중 ({'변형' if use_stocks else '원래'} HAA)...")
    mom_dict = {}
    universe = [t for t in SP500_UNIVERSE if t in monthly.columns] if use_stocks else [BENCHMARK]
    def_universe = [t for t in DEFENSIVE if t in monthly.columns]

    for tk in universe + def_universe + [BENCHMARK]:
        if tk in monthly.columns:
            mom_dict[tk] = calc_momentum(monthly[tk])

    records = []
    dates = monthly.index[12:]

    for i, date in enumerate(dates):
        idx = monthly.index.get_loc(date)
        if idx + 1 >= len(monthly.index):
            continue

        c_mom = canary_mom.get(date, np.nan)
        if pd.isna(c_mom):
            continue

        next_date = monthly.index[idx + 1]

        # ── 공격 / 방어 결정 ──
        if c_mom > 0:
            if use_stocks:
                stock_moms = {
                    tk: mom_dict[tk].loc[date]
                    for tk in universe
                    if tk in mom_dict and date in mom_dict[tk].index and not pd.isna(mom_dict[tk].loc[date])
                }
                chosen = sorted(stock_moms, key=stock_moms.get, reverse=True)[:top_n] or [BENCHMARK]
            else:
                chosen = [BENCHMARK]
            mode = 'attack'
        else:
            def_moms = {
                tk: mom_dict[tk].loc[date]
                for tk in def_universe
                if tk in mom_dict and date in mom_dict[tk].index and not pd.isna(mom_dict[tk].loc[date])
            }
            chosen = [max(def_moms, key=def_moms.get)] if def_moms else ['SHY']
            mode = 'defense'

        # ── 수익률 계산 (균등 비중) ──
        valid = [tk for tk in chosen if tk in monthly.columns
                 and date in monthly.index and next_date in monthly.index
                 and monthly.loc[date, tk] > 0]
        if not valid:
            continue

        port_ret = sum(
            monthly.loc[next_date, tk] / monthly.loc[date, tk] - 1
            for tk in valid
        ) / len(valid)

        records.append({
            'date':     next_date,
            'return':   port_ret,
            'mode':     mode,
            'holdings': ','.join(chosen)
        })

    df = pd.DataFrame(records).set_index('date')
    df['cum_return'] = (1 + df['return']).cumprod()
    return df


# ──────────────────────────────────────────────
# 4. 성과 지표
# ──────────────────────────────────────────────
def performance_stats(df: pd.DataFrame, label: str) -> dict:
    rets = df['return']
    cum  = df['cum_return']
    years = (df.index[-1] - df.index[0]).days / 365.25

    cagr     = cum.iloc[-1] ** (1 / years) - 1
    roll_max = cum.cummax()
    mdd      = ((cum - roll_max) / roll_max).min()
    sharpe   = rets.mean() / rets.std() * np.sqrt(12)
    win_rate = (rets > 0).mean()
    total    = cum.iloc[-1] - 1

    stats = dict(cagr=cagr, mdd=mdd, sharpe=sharpe, win_rate=win_rate, total=total)

    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  CAGR        : {cagr*100:6.2f}%")
    print(f"  MDD         : {mdd*100:6.2f}%")
    print(f"  Sharpe      : {sharpe:6.2f}")
    print(f"  승률        : {win_rate*100:5.1f}%")
    print(f"  누적 수익률 : {total*100:6.1f}%")
    print(f"  공격 비율   : {(df['mode']=='attack').mean()*100:.1f}%")
    return stats


# ──────────────────────────────────────────────
# 5. 시각화
# ──────────────────────────────────────────────
def plot_results(res_mod, res_ori, spy_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'HAA Strategy Backtest Comparison\n'
        'Modified (S&P500 Top Momentum) vs Original (SPY) vs SPY Buy & Hold',
        fontsize=13, fontweight='bold'
    )

    colors = {'mod': '#e74c3c', 'ori': '#2ecc71', 'spy': '#3498db'}

    # (1) 누적 수익률
    ax = axes[0, 0]
    ax.plot(res_mod.index, res_mod['cum_return'], color=colors['mod'], lw=2,   label=f'HAA Modified (Top{TOP_N})')
    ax.plot(res_ori.index, res_ori['cum_return'], color=colors['ori'], lw=2,   label='HAA Original (SPY)')
    ax.plot(spy_df.index,  spy_df['cum_return'],  color=colors['spy'], lw=1.5, label='SPY Buy & Hold', ls='--')
    ax.set_title('Cumulative Return (Growth of $1)')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(); ax.grid(alpha=0.3)

    # (2) 드로우다운
    ax = axes[0, 1]
    for df, label, c in [(res_mod, f'HAA Modified', colors['mod']),
                          (res_ori, 'HAA Original',  colors['ori']),
                          (spy_df,  'SPY',            colors['spy'])]:
        dd = (df['cum_return'] - df['cum_return'].cummax()) / df['cum_return'].cummax() * 100
        ax.fill_between(df.index, dd, 0, alpha=0.35, color=c, label=label)
        ax.plot(df.index, dd, color=c, lw=0.8)
    ax.set_title('Drawdown (%)')
    ax.set_ylabel('Drawdown %')
    ax.legend(); ax.grid(alpha=0.3)

    # (3) 연도별 수익률
    ax = axes[1, 0]
    for df, label, c, offset in [(res_mod, f'HAA Modified', colors['mod'], -0.25),
                                   (res_ori, 'HAA Original',  colors['ori'],  0.00),
                                   (spy_df,  'SPY',            colors['spy'],  0.25)]:
        annual = df['return'].resample('YE').apply(lambda x: (1+x).prod()-1) * 100
        ax.bar(annual.index.year + offset, annual.values, 0.25, label=label, color=c, alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title('Annual Returns (%)')
    ax.set_ylabel('Return %')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    ax.set_xticks(res_mod['return'].resample('YE').mean().index.year)
    ax.set_xticklabels(res_mod['return'].resample('YE').mean().index.year, rotation=45, fontsize=8)

    # (4) 성과 요약 테이블
    ax = axes[1, 1]
    ax.axis('off')

    def gs(df):
        cum = df['cum_return']
        rets = df['return']
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (cum.iloc[-1]**(1/yrs) - 1) * 100
        mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
        sh = rets.mean() / rets.std() * np.sqrt(12)
        wr = (rets > 0).mean() * 100
        atk = (df['mode'] == 'attack').mean() * 100 if 'mode' in df.columns else 100
        return cagr, mdd, sh, wr, atk

    c1 = gs(res_mod); c2 = gs(res_ori); c3 = gs(spy_df)
    rows = [
        ['CAGR (%)',       f'{c1[0]:.1f}%', f'{c2[0]:.1f}%', f'{c3[0]:.1f}%'],
        ['MDD (%)',        f'{c1[1]:.1f}%', f'{c2[1]:.1f}%', f'{c3[1]:.1f}%'],
        ['Sharpe',         f'{c1[2]:.2f}',  f'{c2[2]:.2f}',  f'{c3[2]:.2f}'],
        ['Win Rate (%)',   f'{c1[3]:.1f}%', f'{c2[3]:.1f}%', f'{c3[3]:.1f}%'],
        ['Attack Ratio',   f'{c1[4]:.1f}%', f'{c2[4]:.1f}%', '100.0%'],
    ]
    tbl = ax.table(
        cellText=rows,
        colLabels=['Metric', f'HAA Mod (Top{TOP_N})', 'HAA Original', 'SPY B&H'],
        cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.85]
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
        elif c == 1: cell.set_facecolor('#fde8e8')
        elif c == 2: cell.set_facecolor('#e8fde8')
        elif c == 3: cell.set_facecolor('#e8f0fd')
    ax.set_title('Performance Summary', fontweight='bold')

    plt.tight_layout()
    plt.savefig('haa_backtest_result.png', dpi=150, bbox_inches='tight')
    print("\n차트 저장: haa_backtest_result.png")
    plt.show()


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
if __name__ == '__main__':
    # 1) 데이터
    monthly = download_data()

    # 2) 백테스트
    print("\n[1/3] HAA 변형 (S&P500 Top 모멘텀) 백테스트...")
    res_mod = run_backtest(monthly, use_stocks=True,  top_n=TOP_N)

    print("\n[2/3] HAA 원래 (SPY) 백테스트...")
    res_ori = run_backtest(monthly, use_stocks=False)

    # 3) SPY 단순 보유
    spy_rets = monthly[BENCHMARK].pct_change().dropna()
    spy_df = pd.DataFrame({
        'return':     spy_rets.reindex(res_ori.index),
        'cum_return': (1 + spy_rets).cumprod().reindex(res_ori.index),
        'mode':       'attack'
    })

    # 4) 성과 출력
    print("\n[3/3] 성과 분석...")
    performance_stats(res_mod, f'HAA 변형 — S&P500 Top{TOP_N} 모멘텀')
    performance_stats(res_ori, 'HAA 원래 — SPY')
    performance_stats(spy_df,  'SPY 단순 보유')

    # 5) 차트
    plot_results(res_mod, res_ori, spy_df)
