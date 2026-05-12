import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────
# 1. 설정 및 데이터 경로 (반드시 본인의 Raw 주소로 수정하세요)
# ─────────────────────────────────────────────────────────────────────────
CSV_URL = "https://raw.githubusercontent.com/사용자이름/App-kospi/main/sp500_history.csv"

OFFENSIVE = ['SPY', 'IWM', 'VEA', 'VWO', 'VNQ', 'DBC', 'IEF', 'TLT']
DEFENSIVE = ['SHY', 'IEF', 'BIL']
CANARY    = 'TIP'
TOP_N     = 4

# ─────────────────────────────────────────────────────────────────────────
# 2. 데이터 처리 함수
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_universe_csv(url):
    try:
        df = pd.read_csv(url)
        df['date'] = pd.to_datetime(df['date'])
        df['tickers_list'] = df['tickers'].apply(lambda x: [t.strip() for t in str(x).split(',')])
        return df.sort_values('date')
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        return None


# ✅ 수정: 매달 루프마다 재계산하지 않고 전체 모멘텀 사전 계산 (속도 대폭 향상)
@st.cache_data(show_spinner=False)
def precompute_momentum(_monthly_df):
    """전 종목 Keller식 평균 모멘텀 사전 계산"""
    mom_dict = {}
    for tk in _monthly_df.columns:
        p = _monthly_df[tk].dropna()
        mom = pd.Series(np.nan, index=p.index)
        for i in range(12, len(p)):
            r1  = p.iloc[i] / p.iloc[i-1]  - 1
            r3  = p.iloc[i] / p.iloc[i-3]  - 1
            r6  = p.iloc[i] / p.iloc[i-6]  - 1
            r12 = p.iloc[i] / p.iloc[i-12] - 1
            mom.iloc[i] = (r1 + r3 + r6 + r12) / 4
        mom_dict[tk] = mom
    return mom_dict


# ─────────────────────────────────────────────────────────────────────────
# 3. 백테스트 엔진
# ─────────────────────────────────────────────────────────────────────────
def run_backtest(monthly_df, mom_dict, sp500_df, start_date, replace_spy):
    sim_dates = [d for d in monthly_df.index if d >= pd.to_datetime(start_date)]
    capital = 1.0
    records = []

    for i in range(len(sim_dates) - 1):
        date      = sim_dates[i]
        next_date = sim_dates[i + 1]

        # ✅ 수정: next_date 존재 여부 확인
        if next_date not in monthly_df.index:
            continue

        # 1. 카나리아 필터
        tip_mom = mom_dict.get(CANARY, pd.Series(dtype=float)).get(date, np.nan)
        if pd.isna(tip_mom):
            continue

        # ✅ 수정: spy_replaced 항상 초기화 (방어 모드 NameError 방지)
        spy_replaced = False

        if tip_mom > 0:  # 공격 모드
            off_moms = {
                tk: mom_dict[tk].get(date, np.nan)
                for tk in OFFENSIVE
                if tk in mom_dict and not pd.isna(mom_dict[tk].get(date, np.nan))
            }
            top_assets = sorted(off_moms, key=off_moms.get, reverse=True)[:TOP_N]

            if replace_spy and 'SPY' in top_assets:
                # CSV에서 해당 시점 상위 20개 종목 추출
                past_rows = sp500_df[sp500_df['date'] <= date]
                if not past_rows.empty:
                    top20_univ = past_rows.iloc[-1]['tickers_list'][:20]
                    stock_moms = {
                        tk: mom_dict[tk].get(date, np.nan)
                        for tk in top20_univ
                        if tk in mom_dict and not pd.isna(mom_dict[tk].get(date, np.nan))
                    }
                    if stock_moms:
                        best_stock = max(stock_moms, key=stock_moms.get)
                        top_assets = [best_stock if tk == 'SPY' else tk for tk in top_assets]
                        spy_replaced = True

            chosen, mode = top_assets, 'ATTACK'

        else:  # 방어 모드
            def_moms = {
                tk: mom_dict[tk].get(date, np.nan)
                for tk in DEFENSIVE
                if tk in mom_dict and not pd.isna(mom_dict[tk].get(date, np.nan))
            }
            chosen = [max(def_moms, key=def_moms.get)] if def_moms else ['SHY']
            mode   = 'DEFENSE'
            # spy_replaced = False 는 위에서 이미 초기화됨 ✅

        # 2. 수익률 계산
        rets = []
        for tk in chosen:
            # ✅ 수정: next_date/date 데이터 존재 및 0 나누기 방지
            if (tk in monthly_df.columns
                    and date in monthly_df.index
                    and next_date in monthly_df.index
                    and monthly_df.loc[date, tk] > 0):
                rets.append(monthly_df.loc[next_date, tk] / monthly_df.loc[date, tk] - 1)

        port_ret = sum(rets) / len(rets) if rets else 0
        capital *= (1 + port_ret)

        records.append({
            'date':         next_date,
            'return':       port_ret,
            'cum_return':   capital,
            'mode':         mode,
            'spy_replaced': spy_replaced,
            'holdings':     ', '.join(chosen),
        })

    return pd.DataFrame(records).set_index('date')


# ─────────────────────────────────────────────────────────────────────────
# 4. 성과 지표
# ─────────────────────────────────────────────────────────────────────────
def get_stats(df):
    cum   = df['cum_return']
    rets  = df['return']
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr  = (cum.iloc[-1] ** (1 / years) - 1) * 100
    mdd   = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sh    = rets.mean() / rets.std() * np.sqrt(12)
    wr    = (rets > 0).mean() * 100
    atk   = (df['mode'] == 'ATTACK').mean() * 100 if 'mode' in df.columns else 100.0
    rep   = int(df['spy_replaced'].sum()) if 'spy_replaced' in df.columns else 0
    return dict(cagr=cagr, mdd=mdd, sharpe=sh, win_rate=wr, attack=atk, spy_rep=rep)


# ─────────────────────────────────────────────────────────────────────────
# 5. 바이앤홀드
# ─────────────────────────────────────────────────────────────────────────
def calc_buyhold(monthly_df, ticker, ref_index):
    if ticker not in monthly_df.columns:
        return None
    prices = monthly_df[ticker].reindex(ref_index).dropna()
    rets   = prices.pct_change().dropna()
    cum    = (1 + rets).cumprod()
    return pd.DataFrame({'return': rets, 'cum_return': cum})


# ─────────────────────────────────────────────────────────────────────────
# 6. Streamlit UI
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HAA Alpha Backtester", layout="wide")
st.title("📈 HAA Alpha: SPY Rotation with Top 20 Stocks")
st.markdown("HAA 전략을 기반으로 **SPY 매수 신호 시 S&P 500 시총 상위 20개 중 대장주**를 골라 잡는 백테스트입니다.")

sp500_history = load_universe_csv(CSV_URL)

with st.sidebar:
    st.header("백테스트 설정")
    s_year  = st.slider("시작 연도", 2006, 2020, 2010)
    e_year  = st.slider("종료 연도", 2015, 2026, 2024)
    run_btn = st.button("🚀 백테스트 실행", type="primary", use_container_width=True)

if run_btn and sp500_history is not None:
    with st.spinner("데이터를 수집하고 있습니다. 약 1~2분 소요될 수 있습니다..."):
        # 1. 티커 수집
        all_assets = list(set(OFFENSIVE + DEFENSIVE + [CANARY, 'SPY', 'QQQ']))
        mask = (
            (sp500_history['date'].dt.year >= s_year - 1) &
            (sp500_history['date'].dt.year <= e_year)
        )
        for t_list in sp500_history.loc[mask, 'tickers_list']:
            all_assets.extend(t_list[:20])

        # 2. 데이터 다운로드
        start_dt = (datetime(s_year, 1, 1) - relativedelta(months=13)).strftime('%Y-%m-%d')
        raw = yf.download(
            list(set(all_assets)),
            start=start_dt, end=f"{e_year}-12-31",
            auto_adjust=True, progress=False
        )['Close']
        monthly = raw.resample('ME').last()

    # ✅ 수정: 모멘텀 사전 계산 (백테스트 루프 밖에서 한 번만)
    with st.spinner("모멘텀 계산 중..."):
        mom_dict = precompute_momentum(monthly)

    with st.spinner("백테스트 실행 중..."):
        res_ori = run_backtest(monthly, mom_dict, sp500_history, f"{s_year}-01-01", False)
        res_mod = run_backtest(monthly, mom_dict, sp500_history, f"{s_year}-01-01", True)

    # 바이앤홀드
    spy_bh = calc_buyhold(monthly, 'SPY', res_ori.index)
    qqq_bh = calc_buyhold(monthly, 'QQQ', res_ori.index)

    s_ori = get_stats(res_ori)
    s_mod = get_stats(res_mod)
    s_spy = get_stats(spy_bh) if spy_bh is not None else {}
    s_qqq = get_stats(qqq_bh) if qqq_bh is not None else {}

    # ── 성과 카드 ──
    st.subheader("📊 성과 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("CAGR (변형)", f"{s_mod['cagr']:.2f}%",   f"{s_mod['cagr']  - s_ori['cagr']:+.2f}%")
    col2.metric("MDD (변형)",  f"{s_mod['mdd']:.2f}%",    f"{s_mod['mdd']   - s_ori['mdd']:+.2f}%",  delta_color="inverse")
    col3.metric("Sharpe (변형)", f"{s_mod['sharpe']:.2f}", f"{s_mod['sharpe']- s_ori['sharpe']:+.2f}")

    st.caption(f"SPY 교체 횟수: **{s_mod['spy_rep']}회** / 전체 공격 월수: {int((res_mod['mode']=='ATTACK').sum())}회")

    # ── 전략별 성과 비교 테이블 ──
    st.subheader("📋 전략별 성과 비교")
    compare = {
        '전략':      ['HAA Original', 'HAA Modified', 'SPY B&H', 'QQQ B&H'],
        'CAGR (%)':  [f"{s_ori['cagr']:.2f}%",   f"{s_mod['cagr']:.2f}%",
                      f"{s_spy['cagr']:.2f}%"   if s_spy else '-',
                      f"{s_qqq['cagr']:.2f}%"   if s_qqq else '-'],
        'MDD (%)':   [f"{s_ori['mdd']:.2f}%",    f"{s_mod['mdd']:.2f}%",
                      f"{s_spy['mdd']:.2f}%"    if s_spy else '-',
                      f"{s_qqq['mdd']:.2f}%"    if s_qqq else '-'],
        'Sharpe':    [f"{s_ori['sharpe']:.2f}",  f"{s_mod['sharpe']:.2f}",
                      f"{s_spy['sharpe']:.2f}"  if s_spy else '-',
                      f"{s_qqq['sharpe']:.2f}"  if s_qqq else '-'],
        '승률 (%)':  [f"{s_ori['win_rate']:.1f}%", f"{s_mod['win_rate']:.1f}%",
                      f"{s_spy['win_rate']:.1f}%" if s_spy else '-',
                      f"{s_qqq['win_rate']:.1f}%" if s_qqq else '-'],
    }
    st.dataframe(pd.DataFrame(compare).set_index('전략'), use_container_width=True)

    # ── 누적 수익률 & 드로우다운 차트 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('HAA Original vs Modified vs SPY B&H vs QQQ B&H', fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(res_ori.index, res_ori['cum_return'], color='#2ecc71', lw=2,   label='HAA Original')
    ax.plot(res_mod.index, res_mod['cum_return'], color='#e74c3c', lw=2,   label='HAA Modified')
    if spy_bh is not None:
        ax.plot(spy_bh.index, spy_bh['cum_return'], color='#3498db', lw=1.5, ls='--', label='SPY B&H')
    if qqq_bh is not None:
        ax.plot(qqq_bh.index, qqq_bh['cum_return'], color='#f39c12', lw=1.5, ls='--', label='QQQ B&H')
    ax.set_title('Cumulative Return')
    ax.set_ylabel('Growth of $1')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for df, label, c in [
        (res_ori, 'HAA Original', '#2ecc71'),
        (res_mod, 'HAA Modified', '#e74c3c'),
        *([(spy_bh, 'SPY B&H', '#3498db')] if spy_bh is not None else []),
        *([(qqq_bh, 'QQQ B&H', '#f39c12')] if qqq_bh is not None else []),
    ]:
        dd = (df['cum_return'] - df['cum_return'].cummax()) / df['cum_return'].cummax() * 100
        ax.fill_between(df.index, dd, 0, alpha=0.2, color=c)
        ax.plot(df.index, dd, color=c, lw=1.2, label=label)
    ax.set_title('Drawdown (%)')
    ax.set_ylabel('Drawdown %')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # ── 연도별 수익률 차트 ──
    st.subheader("📅 연도별 수익률")
    fig2, ax2 = plt.subplots(figsize=(14, 5))

    bar_configs = [
        (res_ori, 'HAA Original', '#2ecc71', -0.3),
        (res_mod, 'HAA Modified', '#e74c3c', -0.1),
        *([(spy_bh, 'SPY B&H', '#3498db', 0.1)] if spy_bh is not None else []),
        *([(qqq_bh, 'QQQ B&H', '#f39c12', 0.3)] if qqq_bh is not None else []),
    ]
    for df, label, c, offset in bar_configs:
        annual = df['return'].resample('YE').apply(lambda x: (1+x).prod()-1) * 100
        ax2.bar(annual.index.year + offset, annual.values, 0.18, label=label, color=c, alpha=0.85)

    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title('Annual Returns (%)')
    ax2.legend(); ax2.grid(alpha=0.3, axis='y')
    yrs = res_ori['return'].resample('YE').mean().index.year
    ax2.set_xticks(yrs); ax2.set_xticklabels(yrs, rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    # ── 월별 보유 내역 ──
    with st.expander("월별 리밸런싱 상세 내역 보기"):
        st.dataframe(res_mod.sort_index(ascending=False), use_container_width=True)
