"""
HAA (Hybrid Asset Allocation) 변형 백테스트 - Streamlit 앱
- 원래 HAA: 카나리아(TIP) 양수 → 공격 자산 풀 11개 중 모멘텀 상위 4개 균등 매수
            카나리아(TIP) 음수 → 방어 자산(SHY/IEF/BIL) 중 모멘텀 1위
- 변형 HAA: 공격 자산 4개 중 SPY 포함 시 → S&P500 개별 종목 최고 모멘텀으로 교체
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
OFFENSIVE = ['SPY', 'IWM', 'VEA', 'VWO', 'VNQ', 'DBC', 'IEF', 'TLT']
DEFENSIVE = ['SHY', 'IEF', 'BIL']
CANARY    = 'TIP'
TOP_N     = 4

SP500_UNIVERSE = [
    'AAPL','MSFT','NVDA','GOOGL','META','AVGO','ORCL','ADBE','CRM','AMD',
    'JPM','V','MA','BAC','WFC','GS','MS','BLK','AXP','BRK-B',
    'LLY','UNH','JNJ','ABBV','MRK','TMO','ABT','DHR','BMY','AMGN',
    'AMZN','TSLA','HD','MCD','NKE','SBUX','TGT','LOW','TJX','BKNG',
    'XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY',
    'CAT','HON','UPS','BA','GE','MMM','LMT','RTX','DE','EMR',
    'WMT','PG','KO','PEP','COST','CL','GIS','MO','PM',
    'NEE','DUK','SO','AMT','PLD','CCI','EQIX','SPG','PSA','O',
    'T','VZ','TMUS','NFLX','DIS','CMCSA',
]


# ──────────────────────────────────────────────
# 데이터 & 모멘텀
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def download_data(start, end):
    tickers = list(set(OFFENSIVE + DEFENSIVE + [CANARY] + SP500_UNIVERSE))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    monthly = raw.resample('ME').last()
    monthly = monthly.dropna(axis=1, thresh=len(monthly) // 2)
    return monthly

@st.cache_data(show_spinner=False)
def calc_all_momentum(_monthly):
    all_tickers = OFFENSIVE + DEFENSIVE + [CANARY] + SP500_UNIVERSE
    mom_dict = {}
    for tk in all_tickers:
        if tk in _monthly.columns:
            p = _monthly[tk].dropna()
            mom = pd.Series(np.nan, index=p.index)
            for i in range(12, len(p)):
                r1  = p.iloc[i] / p.iloc[i-1]  - 1
                r3  = p.iloc[i] / p.iloc[i-3]  - 1
                r6  = p.iloc[i] / p.iloc[i-6]  - 1
                r12 = p.iloc[i] / p.iloc[i-12] - 1
                mom.iloc[i] = (r1 + r3 + r6 + r12) / 4
            mom_dict[tk] = mom
    return mom_dict


# ──────────────────────────────────────────────
# 백테스트
# ──────────────────────────────────────────────
def run_backtest(monthly, mom_dict, replace_spy):
    canary_mom   = mom_dict.get(CANARY, pd.Series(dtype=float))
    off_universe = [t for t in OFFENSIVE  if t in mom_dict]
    def_universe = [t for t in DEFENSIVE  if t in mom_dict]
    sp500_univ   = [t for t in SP500_UNIVERSE if t in mom_dict]

    records = []
    for date in monthly.index[12:]:
        idx = monthly.index.get_loc(date)
        if idx + 1 >= len(monthly.index):
            continue
        c_mom = canary_mom.get(date, np.nan)
        if pd.isna(c_mom):
            continue
        next_date = monthly.index[idx + 1]

        if c_mom > 0:
            off_moms = {
                tk: mom_dict[tk].get(date, np.nan)
                for tk in off_universe
                if not pd.isna(mom_dict[tk].get(date, np.nan))
            }
            top4 = sorted(off_moms, key=off_moms.get, reverse=True)[:TOP_N]

            spy_replaced = False
            if replace_spy and 'SPY' in top4:
                sp500_moms = {
                    tk: mom_dict[tk].get(date, np.nan)
                    for tk in sp500_univ
                    if not pd.isna(mom_dict[tk].get(date, np.nan))
                }
                if sp500_moms:
                    best = max(sp500_moms, key=sp500_moms.get)
                    top4 = [best if tk == 'SPY' else tk for tk in top4]
                    spy_replaced = True

            chosen, mode = top4, 'attack'
        else:
            def_moms = {
                tk: mom_dict[tk].get(date, np.nan)
                for tk in def_universe
                if not pd.isna(mom_dict[tk].get(date, np.nan))
            }
            chosen = [max(def_moms, key=def_moms.get)] if def_moms else ['SHY']
            mode, spy_replaced = 'defense', False

        valid = [tk for tk in chosen
                 if tk in monthly.columns and monthly.loc[date, tk] > 0]
        if not valid:
            continue

        port_ret = sum(monthly.loc[next_date, tk] / monthly.loc[date, tk] - 1
                       for tk in valid) / len(valid)
        records.append({
            'date': next_date, 'return': port_ret,
            'mode': mode, 'holdings': ','.join(chosen),
            'spy_replaced': spy_replaced,
        })

    df = pd.DataFrame(records).set_index('date')
    df['cum_return'] = (1 + df['return']).cumprod()
    return df


# ──────────────────────────────────────────────
# 성과 지표
# ──────────────────────────────────────────────
def get_stats(df):
    cum   = df['cum_return']
    rets  = df['return']
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr  = (cum.iloc[-1] ** (1/years) - 1) * 100
    mdd   = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sh    = rets.mean() / rets.std() * np.sqrt(12)
    wr    = (rets > 0).mean() * 100
    atk   = (df['mode'] == 'attack').mean() * 100
    rep   = int(df['spy_replaced'].sum()) if 'spy_replaced' in df.columns else 0
    return dict(cagr=cagr, mdd=mdd, sharpe=sh, win_rate=wr, attack=atk, spy_rep=rep)


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
st.set_page_config(page_title='HAA Backtest', page_icon='📈', layout='wide')
st.title('📈 HAA Strategy Backtest')
st.caption('Original HAA vs Modified HAA (SPY → Top S&P500 Momentum Stock)')

# 사이드바
with st.sidebar:
    st.header('⚙️ 설정')
    start_year = st.slider('시작 연도', 2006, 2020, 2006)
    end_year   = st.slider('종료 연도', 2015, 2026, 2026)
    start = f'{start_year}-01-01'
    end   = f'{end_year}-12-31'
    run_btn = st.button('🚀 백테스트 실행', use_container_width=True)

if run_btn:
    with st.spinner('데이터 다운로드 중...'):
        monthly = download_data(start, end)

    with st.spinner('모멘텀 계산 중... (최초 1회만 시간 소요)'):
        mom_dict = calc_all_momentum(monthly)

    with st.spinner('백테스트 실행 중...'):
        res_ori = run_backtest(monthly, mom_dict, replace_spy=False)
        res_mod = run_backtest(monthly, mom_dict, replace_spy=True)

    s_ori = get_stats(res_ori)
    s_mod = get_stats(res_mod)

    # ── 지표 카드 ──
    st.subheader('📊 성과 요약')
    cols = st.columns(5)
    metrics = [
        ('CAGR', 'cagr', '%'),
        ('MDD',  'mdd',  '%'),
        ('Sharpe', 'sharpe', ''),
        ('승률', 'win_rate', '%'),
        ('공격 비율', 'attack', '%'),
    ]
    for col, (label, key, unit) in zip(cols, metrics):
        v_ori = s_ori[key]
        v_mod = s_mod[key]
        fmt = f'{v_mod:.2f}{unit}' if unit == '' else f'{v_mod:.1f}{unit}'
        delta = v_mod - v_ori
        delta_fmt = f'{delta:+.2f}{unit}' if unit == '' else f'{delta:+.1f}{unit}'
        col.metric(f'{label} (변형)', fmt, delta_fmt)

    st.caption(f"SPY 교체 횟수: **{s_mod['spy_rep']}회** / 전체 공격 월수: {int((res_mod['mode']=='attack').sum())}회")

    # ── 차트 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('HAA Original vs Modified', fontsize=13, fontweight='bold')

    # 누적 수익률
    ax = axes[0]
    ax.plot(res_ori.index, res_ori['cum_return'], color='#2ecc71', lw=2, label='HAA Original')
    ax.plot(res_mod.index, res_mod['cum_return'], color='#e74c3c', lw=2, label='HAA Modified')
    ax.set_title('Cumulative Return')
    ax.set_ylabel('Growth of $1')
    ax.legend(); ax.grid(alpha=0.3)

    # 드로우다운
    ax = axes[1]
    for df, label, c in [(res_ori, 'HAA Original', '#2ecc71'), (res_mod, 'HAA Modified', '#e74c3c')]:
        dd = (df['cum_return'] - df['cum_return'].cummax()) / df['cum_return'].cummax() * 100
        ax.fill_between(df.index, dd, 0, alpha=0.4, color=c, label=label)
        ax.plot(df.index, dd, color=c, lw=0.8)
    ax.set_title('Drawdown (%)')
    ax.set_ylabel('Drawdown %')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # ── 연도별 수익률 ──
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    for df, label, c, offset in [
        (res_ori, 'HAA Original', '#2ecc71', -0.2),
        (res_mod, 'HAA Modified', '#e74c3c',  0.2),
    ]:
        annual = df['return'].resample('YE').apply(lambda x: (1+x).prod()-1) * 100
        ax2.bar(annual.index.year + offset, annual.values, 0.35, label=label, color=c, alpha=0.85)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title('Annual Returns (%)')
    ax2.legend(); ax2.grid(alpha=0.3, axis='y')
    yrs = res_ori['return'].resample('YE').mean().index.year
    ax2.set_xticks(yrs); ax2.set_xticklabels(yrs, rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    # ── 월별 보유 내역 ──
    with st.expander('📋 월별 보유 종목 상세 (변형 HAA)'):
        display_df = res_mod[['mode','holdings','spy_replaced','return']].copy()
        display_df['return'] = (display_df['return'] * 100).round(2).astype(str) + '%'
        display_df.columns = ['모드','보유종목','SPY교체','월수익률']
        st.dataframe(display_df, use_container_width=True)

else:
    st.info('👈 왼쪽 사이드바에서 기간을 설정하고 **백테스트 실행** 버튼을 누르세요.')
