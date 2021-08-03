---
layout: post
toc: true
title: "백테스트로 자동매매 성능 확인하기"
categories: 파이썬으로-업비트-자동매매
sitemap :
changefreq : weekly
priority : 1.0
---

오늘은 Trader의 수익률 성능을 확인하기 위한 백테스트에 대해 소개해 드리겠습니다. 

## Backtest 란
백테스트란 자신의 알고리즘을 과거 데이터에 적용했을 때 얼만큼의 수익을 갖는지 측정하는 것입니다. 대략적으로 자동매매 전략의 성능을 확인할 수 있습니다. 단, 과거 데이터에 대한 수익률은 절대 미래에도 보장되지 않습니다.
또한 백테스트와 현실 거래의 차이를 확인하는 것이 필요합니다.

`Backtester` 등의 파이썬 패키지들이 있지만 직접 구현하는 방법으로 백테스트를 해보겠습니다.
전체 코드는 [Github](https://github.com/LEEMINJOO/only-trading-is-life-way/tree/2021.06.27.ver)에서 확인할 수 있습니다.


## 가상 업비트 만들기

[지난 포스트]({% post_url 2021-03-13-Automatically-earn-1-percent %})에서는 `pyupbit` 패키지를 이용해 실제 업비트 내에 있는 잔고를 이용해 거래 하는 예시를 보여드렸습니다.
하지만 과거 데이터에 대해서 테스트 하기 위해서는 실제 업비트를 활용할 수 없습니다.

업비트 내의 잔고를 확인하고 거래하는 가상 업비트 모듈 `MockUpbit` 를 정의하겠습니다.([전체 코드 한번에 보기](https://github.com/LEEMINJOO/only-trading-is-life-way/blob/2021.06.27.ver/src/mock_upbit.py))

### 잔고 확인하기
```python
class MockUpbit:
    def __init__(
        self,
        seed_money: float,  # 초기 잔고를 입력받습니다.
		ticker: str = "KRW-ETH",  # 이더림움을 기본으로 하겠습니다.
    ):

        # 소유한 현금과 각 코인의 정보를 갖는 
        # `self.balances`를 정의합니다.
        self.balances = {
            "KRW": {
                "currency": "KRW",
                "balance": seed_money,
                "avg_buy_price": 0.,
            },
            ticker: {
                "currency": ticker,
                "balance": 0.,
                "avg_buy_price": 0.,
            },
        }

        self.fee = 0.0005

    def get_balance(
        self,
        ticker,
    ):
        # 잔고 정보를 간단하게 부를 수 있는 
        # 함수 `get_balance`를 정의합니다.
        return self.balances[ticker]["balance"]
```

### 매수하기
`pyupbit` 함수와 동일하게 지정한 금액과 수량으로 코인을 구매하는  `buy_limit_order` 함수를 정의합니다.
현실에서는 실제 거래 수량과 맞아야 체결되지만, 가상으로는 항상 체결된다고 가정합니다.

```python
    def buy_limit_order(
        self,
        ticker: str,
        price: float,
        volume: float,
    ) -> bool:
        if volume <= 0:
            return False

        # 수수료만큼 손해 보기 때문에 수수료만큼 더 부과되도록
        # `total_price`를 할당했습니다.
        total_price = price * volume * (1 + self.fee)

        # 원하는 금액보다 잔고가 많이 있을 때 체결됩니다.
        if self.balances["KRW"]["balance"] < total_price:
            return False

        # 거래 금액만큼 현금 잔고가 줄어듭니다.
        self.balances["KRW"]["balance"] -= total_price

        # 거래한 코인의 평균구매 단가를 계산하고,
        # 추가 매수한만큼 잔고가 증가합니다.
        self.balances[ticker]["avg_buy_price"] = (
            self.balances[ticker]["balance"] * self.balances[ticker]["avg_buy_price"]
            + volume * price
        )
        self.balances[ticker]["balance"] += volume
        self.balances[ticker]["avg_buy_price"] /= self.balances[ticker]["balance"]
        return True
```

### 매도하기
`pyupbit` 함수와 동일하게 지정한 금액과 수량으로 코인을 판매하는  `sell_limit_order` 함수를 정의합니다.
매수와 동일하게 현실에서는 실제 거래 수량과 맞아야 체결되지만, 가상으로는 항상 체결된다고 가정합니다.

```python
    def sell_limit_order(
        self,
        ticker: str,
        price: float,
        volume: float,
    ) -> bool:
        # 수수료만큼 손해 보기 때문에 수수료만큼 더 부과되도록 
        # `total_price`를 할당했습니다.
        total_price = price * volume * (1 - self.fee)

        # 판매하고자 하는 코인의 수량보다 많이 가지고 있을 때만 거래합니다.
        if self.balances[ticker]["balance"] < volume:
            return False

        # 거래한 만큼 잔고를 변화시킵니다.
        self.balances["KRW"]["balance"] += total_price
        self.balances[ticker]["balance"] -= volume
        return True
```

### Trader 적용하기
이렇게 정의한 가상 Upbit 모듈([전체 코드](https://github.com/LEEMINJOO/only-trading-is-life-way/blob/2021.06.27.ver/src/mock_upbit.py))을 통해 지난 포스트에서 만든 BasicTrader를 그대로 사용할 수 있습니다. <br>
(`import` 경로는 Github 최상단으로 하겠습니다.)

```python
>>> from src.trader import BasicTrader
>>> from src.mock_upbit import MockUpbit

>>> seed_money = 1000000
>>> ticker = "KRW-ETH"
>>> upbit = MockUpbit(seed_money)

>>> trader = BasicTrader(upbit=upbit, ticker=ticker)
>>> trader.buy()
>>> trader.krw_balance
>>> trader.ticker_balance
```

<figure class="image" style="center">
  <p>
    <img src="/assets/imgs/upbit/mock_upbit.png" alt="" width="120%">
    <figcaption>가상 업비트를 이용한 Trader</figcaption>
  </p>
</figure>

<figure class="image" style="center">
  <p>
    <img src="/assets/imgs/upbit/mock_upbit_buy.png" alt="" width="120%">
    <figcaption>가상 업비트 거래 모사</figcaption>
  </p>
</figure>

### 주의 사항

가상 업비트 모듈에서는 거래 금액에 관계없이 항상 체결된다고 가정했습니다.
따라서 아래 상황 처럼 현실과 괴리가 있는 금액에 대해서도 처리가 될 수 있습니다.
이런 문제를 보완하기 위한 추가 코드를 아래 [Backteset 하기](#Backteset-하기) 부분에 작성했습니다.

<figure class="image" style="center">
  <p>
    <img src="/assets/imgs/upbit/mock_upbit_warning.png" alt="" width="120%">
    <figcaption>가상 업비트 거래시 주의사항</figcaption>
  </p>
</figure>

## 알고리즘 전략
자동매매 알고리즘이란 Trader가 살것인지, 팔것인지, 얼마에 거래할 것인지 자동을 판단하도록 하는 것입니다.

[지난 포스트]({% post_url 2021-03-13-Automatically-earn-1-percent %})에서는 수동으로 구매하고, 1% 상승 금액으로 판매했습니다.
이번 포스트에서는 Backteset에만 초점을 맞춰 작성하기 위해 가장 최근 종가 금액에 대해 Random으로 사고, 파고, 홀드하는 것으로 전략을 작성하겠습니다.

실제 사용되는 전략을 적용하는 방법은 다음 포스트에서 설명드리겠습니다.

### 랜덤 매매 전략 작성하기

지난번에 작성한 BasicTrader에 `check_market_status_price`함수를 추가 작성하겠습니다. 
전체 코드는 [Github](https://github.com/LEEMINJOO/only-trading-is-life-way/blob/2021.06.27.ver/src/trader/basic_trader.py)에서 확인할 수 있습니다.

가장 최근 정보의 거래내역이 입력되도록 했습니다.

```python
import random 

def check_market_status_price(self, df):
	# 랜덤으로 사고 팔 것인지 결정합니다.
	status = ["buy", "sell", "none"]
	status = random.sample(status, 1)[0]
	# 거래 금액은 최근의 종가로 합니다.
	price = df["close"].iloc[-1]
	return status, price
```

반환된 상태와 금액을 아래와 같이 이용할 수 있습니다.

```python
status, price = trader.check_market_status_price(data)

if status == “buy”:
    trader.buy(ticker, price, volume)
elif status == “sell”:
    trader.sell(ticker, price, volume)
```

<figure class="image" style="center">
  <p>
    <img src="/assets/imgs/upbit/trader_check_staus_random.png" alt="" width="120%">
    <figcaption>랜덤 전략</figcaption>
  </p>
</figure>

## Backteset 하기

### Backteset 과정
한 시점의 데이터가 하나씩 입력될 때 `Trader`가 각 시점마다 사고 팔지 판단하는 상황을 가정하겠습니다.

1. t 시점의 데이터를 계속해서 trader에 입력해줍니다.
2. 입력된 데이터를 가지고 trader는 살지, 팔지, 그대로 있을지를 판단합니다.
3. 사거나 파는 결정을 했다면 거래 금액도 trader가 결정하도록 합니다.
4. 결정된 금액이 t + 1 시점의 저가(low)와 고가(high) 사이에 있는 경우 거래합니다.

### 거래 금액 제한하기
앞선 [주의 사항](#주의-사항)에서 언급한 문제를 해결하기 위해 `4.`번의 제약을 추가했습니다.<br>
`2.`번과 `3.`번 과정은 앞에서 정의한 Trader를 이용할 수 있습니다.

```python
def check_available_bought_price(
    price,  # 매수 가격
    low,    # 다음 시점 저가
    high,   # 다음 시점 고가
):
    assert low <= high
    # 저가 보다 매수 가격이 큰 경우 거래 가능
    if low < price:
        # 고가 보다 매수 가격이 큰 경우 고가로 거래
        return True, min(high, price)
    return False, None


def check_available_sold_price(
    price,  # 매도 가격
    low,    # 다음 시점 저가
    high,   # 다음 시점 고가
):
    assert low <= high
    # 고가 보다 매도 가격이 작은 경우 거래 가능
    if price < high:
        # 저가 보다 매도 가격이 작은 경우 저가로 거래
        return True, max(price, low)
    return False, None
```

### Backteset 코드
[Backteset 과정](#Backteset-과정)의 내용을 파이썬 코드로 작성하겠습니다.[[전체 코드 보기](https://github.com/LEEMINJOO/only-trading-is-life-way/blob/2021.06.27.ver/src/backtesting.py)]

최종 성능으로 ROI(Return on Investment, 투자 수익률)를 반환하겠습니다.

```python
def backtesting(
    test_data: pd.DataFrame,
    seed_money: float = 1000000,
    ticker: str = "KRW-EHT",
) -> float:

    upbit = MockUpbit(seed_money, ticker)
    trader = BasicTrader(upbit=upbit, ticker=ticker)

    # 모든 시점에 대해 for-loop을 돕니다.
    for t in range(test_data.shape[0] - 1):
        # t 시점의 데이터를 불러옵니다.
        data = test_data.iloc[t: t+1]

        # 입력된 t 시점의 데이터를 바탕으로
        # 살지, 팔지, 그대로 있을지와 거래 금액을 결정합니다.
        status, price = trader.check_market_status_price(data)

        # t + 1 시점의 데이터 중 저가와 고가를 추출합니다.
        next_data = test_data.iloc[t+1]
        low, high = next_data["low"], next_data["high"]

        # 매수 상황
        if status == "buy":
            # 거래 금액 제약을 확인합니다.
            available, price = check_available_bought_price(price, low, high)
            if available:
                trader.buy(price)

        # 매도 상황
        elif status == "sell":
            # 거래 금액 제약을 확인합니다.
            available, price = check_available_sold_price(price, low, high)
            if available:
                trader.sell(price)

    # 최근 코인 가격으로 총 자산을 계산합니다.
    recent_ticker_price = test_data["close"].iloc[-1]
    total_balance = (
        trader.krw_balance
        + trader.ticker_balance * recent_ticker_price
    )

    # 초기 자본금 대비 수익률을 계산합니다.
    ROI = ((total_balance - seed_money) / seed_money) * 100
    print(ROI, "% !!!!!")
	return ROI
```

### 랜덤 매매 전략 Backteset 결과

랜덤 매매 전략을 이더리움의 최근 300일 데이터에 대해 백테스트 해봤습니다. <br>
결과 ROI는 2%로 거의 초기 투자금과 비슷한 결과를 얻었습니다.  

```python
from backtesting import backtesting
import pyupbit

ticker = "KRW-ETH"
data = pyupbit.get_ohlcv(ticker=ticker, count=300)
backtesting(test_data=data)
```

<figure class="image" style="center">
  <p>
    <img src="/assets/imgs/upbit/backtesting.png" alt="" width="120%">
  </p>
</figure>

<figure class="image" style="center">
  <p>
    <img src="/assets/imgs/upbit/backtesting_result.png" alt="" width="120%">
    <figcaption>랜덤 전략 백테스트 결과</figcaption>
  </p>
</figure>


이번 포스트에서 Backteset에 대해 설명드렸습니다.
중간 중간 설명된 현실 거래와의 차이로 인해 실제 적용하면 결과가 다를 수 있습니다.

다음 포스트에서는 다른 알고리즘 전략에 대해 설명 드리고, 이번 포스트에서 만든 Backteset을 적용해 보겠습니다.
