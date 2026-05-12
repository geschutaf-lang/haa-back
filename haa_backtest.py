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
CSV_URL = "https://raw.githubusercontent.com/fja05680/sp500/refs/heads/master/S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv"

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
        # tickers 컬럼의 문자열을 리스트로 변환
        df['tickers_list'] = df['tickers'].apply(lambda x: [t.strip() for t in str(x).split(',')])
        return df.sort_values('date')
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        return None

def get_haa_momentum(prices):
    """표준 HAA 모멘텀 스코어: (1m + 3m + 6m + 12m) / 4"""
    if len(prices) < 13: return np.nan
    # 현재가 대비 과거 n개월 전 종가 수익률
    r1  = prices.iloc[-1] / prices.iloc[-2] - 1
    r3  = prices.iloc[-1] / prices.iloc[-4] - 1
    r6  = prices.iloc[-1] / prices.iloc[-7] - 1
    r12 = prices.iloc[-1] / prices.iloc[-13] - 1
    return (r1 + r3 + r6 + r12) / 4

# ─────────────────────────────────────────────────────────────────────────
# 3. 백테스트 엔진
# ─────────────────────────────────────────────────────────────────────────
def run_backtest(monthly_df, sp500_df, start_date, replace_spy):
    sim_dates = [d for d in monthly_df.index if d >= pd.to_datetime(start_date)]
    capital = 1.0
    records = []
    
    for i in range(len(sim_dates) - 1):
        date = sim_dates[i]
        next_date = sim_dates[i+1]
        
        # 1. 카나리아 필터 (TIP 모멘텀 측정)
        tip_prices = monthly_df[CANARY].loc[:date]
        tip_mom = get_haa_momentum(tip_prices)
        
        if tip_mom > 0: # 공격 모드
            off_moms = {tk: get_haa_momentum(monthly_df[tk].loc[:date]) 
                        for tk in OFFENSIVE if tk in monthly_df.columns}
            top_assets = sorted(off_moms, key=off_moms.get, reverse=True)[:TOP_N]
            
            spy_replaced = False
            # 💡 SPY가 선정되었을 때 상위 20개 종목 중 1위로 교체
            if replace_spy and 'SPY' in top_assets:
                # CSV에서 해당 시점의 상위 20개 종목 추출
                target_row = sp500_df[sp500_df['date'] <= date].iloc[-1]
                top20_univ = target_row['tickers_list'][:20]
                
                # 상위 20개 중 모멘텀 최고 종목 선정
                stock_moms = {tk: get_haa_momentum(monthly_df[tk].loc[:date]) 
                              for tk in top20_univ if tk in monthly_df.columns}
                if stock_moms:
                    best_stock = max(stock_moms, key=stock_moms.get)
                    top_assets = [best_stock if tk == 'SPY' else tk for tk in top_assets]
                    spy_replaced = True
            
            chosen, mode = top_assets, 'ATTACK'
        else: # 방어 모드
            def_moms = {tk: get_haa_momentum(monthly_df[tk].loc[:date]) 
                        for tk in DEFENSIVE if tk in monthly_df.columns}
            chosen, mode = [max(def_moms, key=def_moms.get)] if def_moms else ['SHY'], 'DEFENSE'

        # 2. 수익률 계산 및 자산 업데이트
        rets = [monthly_df.loc[next_date, tk] / monthly_df.loc[date, tk] - 1 
                for tk in chosen if tk in monthly_df.columns]
        port_ret = sum(rets) / len(rets) if rets else 0
        capital *= (1 + port_ret)
        
        records.append({
            'date': next_date, 
            'return': port_ret, 
            'cum_return': capital, 
            'mode': mode, 
            'spy_replaced': spy_replaced,
            'holdings': ", ".join(chosen)
        })

    return pd.DataFrame(records).set_index('date')

# ─────────────────────────────────────────────────────────────────────────
# 4. 스트림릿 UI 및 실행 레이어
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HAA Alpha Backtester", layout="wide")
st.title("📈 HAA Alpha: SPY Rotation with Top 20 Stocks")
st.markdown("HAA 전략을 기반으로 **SPY 매수 신호 시 S&P 500 시총 상위 20개 중 대장주**를 골라 잡는 무결점 백테스트입니다.")

sp500_history = load_universe_csv(CSV_URL)

with st.sidebar:
    st.header("백테스트 설정")
    s_year = st.slider("시작 연도", 2006, 2020, 2010)
    e_year = st.slider("종료 연도", 2015, 2026, 2024)
    run_btn = st.button("🚀 백테스트 실행", type="primary", use_container_width=True)

if run_btn and sp500_history is not None:
    with st.spinner("데이터를 수집하고 있습니다. 종목 수가 많아 약 1~2분 정도 소요될 수 있습니다..."):
        # 1. 분석에 필요한 모든 티커 리스트업
        all_assets = list(set(OFFENSIVE + DEFENSIVE + [CANARY]))
        # CSV 전체 역사에서 '상위 20개'에 한 번이라도 이름을 올린 모든 종목 추가
        mask = (sp500_history['date'].dt.year >= s_year - 1) & (sp500_history['date'].dt.year <= e_year)
        for t_list in sp500_history.loc[mask, 'tickers_list']:
            all_assets.extend(t_list[:20])
        
        # 2. 데이터 다운로드 (1년치 여유 데이터 포함)
        start_dt = (datetime(s_year, 1, 1) - relativedelta(months=13)).strftime('%Y-%m-%d')
        raw = yf.download(list(set(all_assets)), start=start_dt, end=f"{e_year}-12-31", progress=False)['Close']
        monthly = raw.resample('ME').last()
        
        # 3. 백테스트 실행 (오리지널 HAA vs 변형 HAA)
        res_ori = run_backtest(monthly, sp500_history, f"{s_year}-01-01", False)
        res_mod = run_backtest(monthly, sp500_history, f"{s_year}-01-01", True)

    # 📊 결과 요약 카드
    def get_stats(df):
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (df['cum_return'].iloc[-1] ** (1/years) - 1) * 100
        mdd = ((df['cum_return'] - df['cum_return'].cummax()) / df['cum_return'].cummax()).min() * 100
        sharpe = (df['return'].mean() / df['return'].std()) * np.sqrt(12)
        return cagr, mdd, sharpe

    cagr_m, mdd_m, sh_m = get_stats(res_mod)
    cagr_o, mdd_o, sh_o = get_stats(res_ori)

    col1, col2, col3 = st.columns(3)
    col1.metric("연평균 수익률(CAGR)", f"{cagr_m:.2f}%", f"{cagr_m - cagr_o:+.2f}%")
    col2.metric("최대 낙폭(MDD)", f"{mdd_m:.2f}%", f"{mdd_m - mdd_o:+.2f}%", delta_color="inverse")
    col3.metric("샤프 지수 (Sharpe)", f"{sh_m:.2f}", f"{sh_m - sh_o:+.2f}")

    # 📈 누적 수익률 차트
    st.subheader("누적 수익률 비교 (HAA Original vs Alpha)")
    comparison = pd.DataFrame({
        'Original HAA': res_ori['cum_return'],
        'Modified Alpha': res_mod['cum_return']
    })
    st.line_chart(comparison)

    # 📋 상세 기록
    with st.expander("월별 리밸런싱 상세 내역 보기"):
        st.dataframe(res_mod.sort_index(ascending=False), use_container_width=True)
