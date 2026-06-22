"""Tests for KalshiSource fetch/update logic."""

from datetime import date, datetime, timezone
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from _fb_types import SourceQuestionBank
from _schemas import KalshiFetchFrame, QuestionFrame, ResolutionFrame
from resolve._impute import impute_missing_forecasts
from resolve._prepare import check_and_prepare_forecast_file, set_resolution_dates
from resolve.explode_question_set import explode_question_set
from resolve.resolve_all import resolve_all
from sources.kalshi import KalshiSource
from sources.registry import SOURCES

from .conftest import (
    make_kalshi_api_market,
    make_kalshi_candlestick,
    make_kalshi_event,
    make_kalshi_fetch_df,
    make_question_df,
    make_question_set_df,
    make_resolution_df,
)


def _ts(year, month, day, hour=12):
    """Return the unix timestamp (seconds) for a UTC datetime."""
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp())


# ---------------------------------------------------------------------------
# _get_resolved_market_value (pure, no mocking)
# ---------------------------------------------------------------------------


class TestGetResolvedMarketValue:
    """Tests for KalshiSource._get_resolved_market_value static method."""

    def test_yes_resolution(self):
        """'yes' result returns 1."""
        assert KalshiSource._get_resolved_market_value(make_kalshi_api_market(result="yes")) == 1

    def test_no_resolution(self):
        """'no' result returns 0."""
        assert KalshiSource._get_resolved_market_value(make_kalshi_api_market(result="no")) == 0

    def test_empty_result_is_nan(self):
        """Empty result returns NaN."""
        assert np.isnan(KalshiSource._get_resolved_market_value(make_kalshi_api_market(result="")))

    def test_scalar_result_is_nan(self):
        """A non yes/no result returns NaN."""
        result = KalshiSource._get_resolved_market_value(make_kalshi_api_market(result="scalar"))
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# _is_resolved / _series_ticker (pure)
# ---------------------------------------------------------------------------


class TestMarketHelpers:
    """Tests for small Kalshi static helpers."""

    def test_is_resolved_terminal_statuses(self):
        """Terminal statuses are treated as resolved."""
        for status in ["settled", "finalized", "determined"]:
            assert KalshiSource._is_resolved(make_kalshi_api_market(status=status)) is True

    def test_is_resolved_non_terminal(self):
        """Active/closed markets are not yet resolved."""
        for status in ["active", "closed", "initialized"]:
            assert KalshiSource._is_resolved(make_kalshi_api_market(status=status)) is False

    def test_series_ticker(self):
        """Series ticker is the prefix before the first dash."""
        assert KalshiSource._series_ticker("KXWCSPREAD-26JUN18CANQAT-CAN6") == "KXWCSPREAD"

    def test_resolution_criteria_joins_rules(self):
        """Primary and secondary rules are joined, empties dropped."""
        market = make_kalshi_api_market(rules_primary="Primary.", rules_secondary="Secondary.")
        assert KalshiSource._resolution_criteria(market) == "Primary. Secondary."

    def test_resolution_criteria_na_when_empty(self):
        """No rules yields 'N/A'."""
        market = make_kalshi_api_market(rules_primary="", rules_secondary="")
        assert KalshiSource._resolution_criteria(market) == "N/A"

    def test_resolution_datetime_prefers_settlement(self):
        """settlement_ts is preferred over expiration/close."""
        market = make_kalshi_api_market(settlement_ts="2026-01-13T05:00:00Z")
        assert KalshiSource._resolution_datetime(market).startswith("2026-01-13")


# ---------------------------------------------------------------------------
# _build_resolution_df (mock _get_market_candlesticks)
# ---------------------------------------------------------------------------


class TestBuildResolutionDf:
    """Tests for KalshiSource._build_resolution_df."""

    @patch.object(KalshiSource, "_get_market_candlesticks")
    def test_already_up_to_date(self, mock_candles, kalshi_source, freeze_today):
        """Skips API call if existing data covers through yesterday."""
        freeze_today(date(2026, 1, 15))
        existing = make_resolution_df(
            [
                {"id": "KXTEST-001", "date": "2024-06-01", "value": 0.5},
                {"id": "KXTEST-001", "date": "2026-01-14", "value": 0.6},
            ]
        )
        market = make_kalshi_api_market()
        result = kalshi_source._build_resolution_df(
            market=market, market_info_resolution_datetime="N/A", existing_df=existing
        )

        assert result.equals(existing)
        mock_candles.assert_not_called()

    @patch.object(KalshiSource, "_get_market_candlesticks")
    def test_basic_unresolved_market(self, mock_candles, kalshi_source, freeze_today):
        """Builds a valid time series from candlesticks for an unresolved market."""
        freeze_today(date(2026, 1, 15))
        mock_candles.return_value = [
            make_kalshi_candlestick(_ts(2026, 1, 10), close_dollars="0.40"),
            make_kalshi_candlestick(_ts(2026, 1, 12), close_dollars="0.60"),
        ]
        result = kalshi_source._build_resolution_df(
            market=make_kalshi_api_market(), market_info_resolution_datetime="N/A", existing_df=None
        )

        assert result is not None
        assert (result["id"] == "KXTEST-001").all()
        ResolutionFrame.validate(result)
        # Candles shifted back one day (09, 11), forward-filled: 09, 10, 11, 12, 13, 14
        assert len(result) >= 5

    @patch.object(KalshiSource, "_get_market_candlesticks")
    def test_empty_candles_returns_none(self, mock_candles, kalshi_source, freeze_today):
        """No candlesticks returns None."""
        freeze_today(date(2026, 1, 15))
        mock_candles.return_value = []
        result = kalshi_source._build_resolution_df(
            market=make_kalshi_api_market(), market_info_resolution_datetime="N/A", existing_df=None
        )
        assert result is None

    @patch.object(KalshiSource, "_get_market_candlesticks")
    def test_no_trade_candles_returns_none(self, mock_candles, kalshi_source, freeze_today):
        """Candlesticks with no trades (empty price) return None."""
        freeze_today(date(2026, 1, 15))
        mock_candles.return_value = [
            make_kalshi_candlestick(_ts(2026, 1, 10)),
            make_kalshi_candlestick(_ts(2026, 1, 12)),
        ]
        result = kalshi_source._build_resolution_df(
            market=make_kalshi_api_market(), market_info_resolution_datetime="N/A", existing_df=None
        )
        assert result is None

    @patch.object(KalshiSource, "_get_market_candlesticks")
    def test_forward_fills_gaps(self, mock_candles, kalshi_source, freeze_today):
        """Missing dates between candlesticks are forward-filled."""
        freeze_today(date(2026, 1, 15))
        mock_candles.return_value = [
            make_kalshi_candlestick(_ts(2026, 1, 10), close_dollars="0.30"),
            make_kalshi_candlestick(_ts(2026, 1, 14), close_dollars="0.80"),
        ]
        result = kalshi_source._build_resolution_df(
            market=make_kalshi_api_market(), market_info_resolution_datetime="N/A", existing_df=None
        )

        dates_in_df = pd.to_datetime(result["date"]).dt.date.tolist()
        assert date(2026, 1, 11) in dates_in_df
        assert date(2026, 1, 12) in dates_in_df
        assert date(2026, 1, 13) in dates_in_df

    @patch.object(KalshiSource, "_get_market_candlesticks")
    def test_resolved_truncates_at_resolution(self, mock_candles, kalshi_source, freeze_today):
        """Resolved market: data truncated at resolution date, final row has resolved value."""
        freeze_today(date(2026, 1, 15))
        mock_candles.return_value = [
            make_kalshi_candlestick(_ts(2026, 1, 10), close_dollars="0.40"),
            make_kalshi_candlestick(_ts(2026, 1, 12), close_dollars="0.60"),
            make_kalshi_candlestick(_ts(2026, 1, 14), close_dollars="0.90"),
        ]
        market = make_kalshi_api_market(status="finalized", result="yes")
        result = kalshi_source._build_resolution_df(
            market=market,
            market_info_resolution_datetime="2026-01-13T12:00:00+00:00",
            existing_df=None,
        )

        assert result is not None
        last_date = pd.to_datetime(result["date"].iloc[-1]).date()
        assert last_date == date(2026, 1, 13)
        assert float(result["value"].iloc[-1]) == 1.0
        all_dates = pd.to_datetime(result["date"]).dt.date
        assert all(d <= date(2026, 1, 13) for d in all_dates)

    @patch.object(KalshiSource, "_get_market_candlesticks")
    def test_resolved_void_nan_last_row(self, mock_candles, kalshi_source, freeze_today):
        """Void resolution (empty result) on a terminal market: last row is NaN."""
        freeze_today(date(2026, 1, 15))
        mock_candles.return_value = [
            make_kalshi_candlestick(_ts(2026, 1, 10), close_dollars="0.40"),
            make_kalshi_candlestick(_ts(2026, 1, 12), close_dollars="0.60"),
        ]
        market = make_kalshi_api_market(status="finalized", result="")
        result = kalshi_source._build_resolution_df(
            market=market,
            market_info_resolution_datetime="2026-01-13T12:00:00+00:00",
            existing_df=None,
        )

        assert result is not None
        assert np.isnan(float(result["value"].iloc[-1]))

    @patch.object(KalshiSource, "_get_market_candlesticks")
    def test_filters_future_candles(self, mock_candles, kalshi_source, freeze_today):
        """Candlesticks dated after yesterday are excluded (after the one-day shift back)."""
        freeze_today(date(2026, 1, 15))
        # After the one-day shift these land on 01-14 (kept) and 01-15 (today, excluded).
        mock_candles.return_value = [
            make_kalshi_candlestick(_ts(2026, 1, 15), close_dollars="0.50"),
            make_kalshi_candlestick(_ts(2026, 1, 16), close_dollars="0.90"),
        ]
        result = kalshi_source._build_resolution_df(
            market=make_kalshi_api_market(), market_info_resolution_datetime="N/A", existing_df=None
        )

        assert result is not None
        all_dates = pd.to_datetime(result["date"]).dt.date
        assert all(d <= date(2026, 1, 14) for d in all_dates)
        assert date(2026, 1, 15) not in all_dates.tolist()

    @patch.object(KalshiSource, "_get_market_candlesticks")
    def test_candle_dates_shifted_back_one_day(self, mock_candles, kalshi_source, freeze_today):
        """ET-anchored candles are attributed to the prior calendar day (matches polymarket)."""
        freeze_today(date(2026, 1, 15))
        mock_candles.return_value = [
            make_kalshi_candlestick(_ts(2026, 1, 12), close_dollars="0.55"),
        ]
        result = kalshi_source._build_resolution_df(
            market=make_kalshi_api_market(), market_info_resolution_datetime="N/A", existing_df=None
        )

        result_dates = pd.to_datetime(result["date"]).dt.date
        # candle whose end_period_ts falls on 2026-01-12 is labeled 2026-01-11
        assert result_dates.min() == date(2026, 1, 11)
        first_val = result.loc[result_dates == date(2026, 1, 11), "value"].iloc[0]
        assert float(first_val) == 0.55


# ---------------------------------------------------------------------------
# _call_search_endpoint (mock requests.get)
# ---------------------------------------------------------------------------


class TestCallSearchEndpoint:
    """Tests for KalshiSource._call_search_endpoint."""

    def _mock_response(self, events, cursor=None):
        resp = Mock()
        resp.ok = True
        resp.json.return_value = {"events": events, "cursor": cursor}
        resp.raise_for_status = Mock()
        return resp

    @patch("sources.kalshi.requests.get")
    def test_basic_returns_qualifying_tickers(self, mock_get, kalshi_source):
        """Returns tickers for markets meeting all criteria."""
        events = [
            make_kalshi_event(
                category="Economics",
                markets=[
                    make_kalshi_api_market(ticker="A"),
                    make_kalshi_api_market(ticker="B"),
                ],
            )
        ]
        mock_get.return_value = self._mock_response(events)
        ids, cursor = kalshi_source._call_search_endpoint(min_resolution_date=date(2026, 1, 25))
        assert ids == {"A", "B"}
        assert cursor is None

    @patch("sources.kalshi.requests.get")
    def test_filters_non_binary(self, mock_get, kalshi_source):
        """Scalar markets are excluded."""
        events = [
            make_kalshi_event(
                category="Economics",
                markets=[
                    make_kalshi_api_market(ticker="bin", market_type="binary"),
                    make_kalshi_api_market(ticker="scal", market_type="scalar"),
                ],
            )
        ]
        mock_get.return_value = self._mock_response(events)
        ids, _ = kalshi_source._call_search_endpoint(min_resolution_date=date(2026, 1, 25))
        assert ids == {"bin"}

    @patch("sources.kalshi.requests.get")
    def test_filters_low_volume(self, mock_get, kalshi_source):
        """Markets with volume below the floor are excluded."""
        events = [
            make_kalshi_event(
                category="Economics",
                markets=[
                    make_kalshi_api_market(ticker="low", volume_fp="100.00"),
                    make_kalshi_api_market(ticker="ok", volume_fp="10000.00"),
                ],
            )
        ]
        mock_get.return_value = self._mock_response(events)
        ids, _ = kalshi_source._call_search_endpoint(min_resolution_date=date(2026, 1, 25))
        assert ids == {"ok"}

    @patch("sources.kalshi.requests.get")
    def test_filters_low_open_interest(self, mock_get, kalshi_source):
        """Markets with open interest below the floor are excluded."""
        events = [
            make_kalshi_event(
                category="Economics",
                markets=[
                    make_kalshi_api_market(ticker="low", open_interest_fp="10.00"),
                    make_kalshi_api_market(ticker="ok", open_interest_fp="2000.00"),
                ],
            )
        ]
        mock_get.return_value = self._mock_response(events)
        ids, _ = kalshi_source._call_search_endpoint(min_resolution_date=date(2026, 1, 25))
        assert ids == {"ok"}

    @patch("sources.kalshi.requests.get")
    def test_filters_close_before_min_resolution(self, mock_get, kalshi_source):
        """Markets closing before the minimum resolution date are excluded."""
        events = [
            make_kalshi_event(
                category="Economics",
                markets=[
                    make_kalshi_api_market(ticker="soon", close_time="2026-01-20T00:00:00Z"),
                    make_kalshi_api_market(ticker="ok", close_time="2026-03-01T00:00:00Z"),
                ],
            )
        ]
        mock_get.return_value = self._mock_response(events)
        ids, _ = kalshi_source._call_search_endpoint(min_resolution_date=date(2026, 1, 25))
        assert ids == {"ok"}

    @patch("sources.kalshi.requests.get")
    def test_filters_close_after_max_resolution(self, mock_get, kalshi_source):
        """Markets closing after the maximum resolution date (e.g. 2099 novelty markets)."""
        events = [
            make_kalshi_event(
                category="Economics",
                markets=[
                    make_kalshi_api_market(ticker="ok", close_time="2026-06-01T00:00:00Z"),
                    make_kalshi_api_market(ticker="far", close_time="2099-01-01T00:00:00Z"),
                ],
            )
        ]
        mock_get.return_value = self._mock_response(events)
        ids, _ = kalshi_source._call_search_endpoint(
            min_resolution_date=date(2026, 1, 25),
            max_resolution_date=date(2028, 1, 1),
        )
        assert ids == {"ok"}

    @patch("sources.kalshi.requests.get")
    def test_trending_non_category_included(self, mock_get, kalshi_source):
        """A market outside target categories is included when trending."""
        events = [
            make_kalshi_event(
                category="Sports",
                markets=[make_kalshi_api_market(ticker="hot", volume_24h_fp="50000.00")],
            )
        ]
        mock_get.return_value = self._mock_response(events)
        ids, _ = kalshi_source._call_search_endpoint(min_resolution_date=date(2026, 1, 25))
        assert ids == {"hot"}

    @patch("sources.kalshi.requests.get")
    def test_non_category_non_trending_excluded(self, mock_get, kalshi_source):
        """A non-trending market outside target categories is excluded."""
        events = [
            make_kalshi_event(
                category="Sports",
                markets=[make_kalshi_api_market(ticker="cold", volume_24h_fp="10.00")],
            )
        ]
        mock_get.return_value = self._mock_response(events)
        ids, _ = kalshi_source._call_search_endpoint(min_resolution_date=date(2026, 1, 25))
        assert ids == set()

    @patch("sources.kalshi.requests.get")
    def test_cursor_passed_through(self, mock_get, kalshi_source):
        """The next-page cursor is returned and an incoming cursor is sent in params."""
        mock_get.return_value = self._mock_response([], cursor="next_page")
        _, cursor = kalshi_source._call_search_endpoint(
            min_resolution_date=date(2026, 1, 25), cursor="cur1"
        )
        assert cursor == "next_page"
        assert mock_get.call_args.kwargs["params"]["cursor"] == "cur1"


# ---------------------------------------------------------------------------
# _get_market / _get_market_candlesticks (mock requests.get)
# ---------------------------------------------------------------------------


class TestGetMarket:
    """Tests for KalshiSource._get_market."""

    @patch("sources.kalshi.time.sleep")
    @patch("sources.kalshi.requests.get")
    def test_returns_market_object(self, mock_get, mock_sleep, kalshi_source):
        """Unwraps and returns the 'market' object."""
        resp = Mock()
        resp.ok = True
        resp.json.return_value = {"market": make_kalshi_api_market(ticker="KXTEST-001")}
        mock_get.return_value = resp

        result = kalshi_source._get_market("KXTEST-001")
        assert result["ticker"] == "KXTEST-001"


class TestGetMarketCandlesticks:
    """Tests for KalshiSource._get_market_candlesticks."""

    @patch("sources.kalshi.time.sleep")
    @patch("sources.kalshi.requests.get")
    def test_builds_series_url_and_returns_candles(
        self, mock_get, mock_sleep, kalshi_source, freeze_today
    ):
        """Uses the series-derived URL and returns the candlesticks list."""
        freeze_today(date(2026, 1, 15))
        resp = Mock()
        resp.ok = True
        resp.json.return_value = {
            "candlesticks": [make_kalshi_candlestick(_ts(2026, 1, 10), close_dollars="0.5")]
        }
        mock_get.return_value = resp

        result = kalshi_source._get_market_candlesticks("KXWCSPREAD-26JUN18CANQAT-CAN6")
        assert len(result) == 1
        url = mock_get.call_args[0][0]
        assert "/series/KXWCSPREAD/markets/KXWCSPREAD-26JUN18CANQAT-CAN6/candlesticks" in url
        assert mock_get.call_args.kwargs["params"]["period_interval"] == 1440

    @patch("sources.kalshi.time.sleep")
    @patch("sources.kalshi.requests.get")
    def test_empty_candles(self, mock_get, mock_sleep, kalshi_source, freeze_today):
        """Missing candlesticks key returns empty list."""
        freeze_today(date(2026, 1, 15))
        resp = Mock()
        resp.ok = True
        resp.json.return_value = {"ticker": "KXTEST-001"}
        mock_get.return_value = resp
        assert kalshi_source._get_market_candlesticks("KXTEST-001") == []


# ---------------------------------------------------------------------------
# fetch() (mock _search_markets)
# ---------------------------------------------------------------------------


class TestFetch:
    """Tests for KalshiSource.fetch."""

    @patch.object(KalshiSource, "_search_markets")
    def test_basic_fetch(self, mock_search, kalshi_source):
        """Returns sorted KalshiFetchFrame with correct tickers."""
        mock_search.return_value = {"id_b", "id_a", "id_c"}
        dff = kalshi_source.fetch()

        assert dff["id"].tolist() == ["id_a", "id_b", "id_c"]
        KalshiFetchFrame.validate(dff)

    @patch.object(KalshiSource, "_search_markets")
    def test_empty_results(self, mock_search, kalshi_source):
        """Empty search returns empty valid frame."""
        mock_search.return_value = set()
        dff = kalshi_source.fetch()

        assert len(dff) == 0
        KalshiFetchFrame.validate(dff)


# ---------------------------------------------------------------------------
# update() (mock _get_market + _build_resolution_df)
# ---------------------------------------------------------------------------


class TestUpdate:
    """Tests for KalshiSource.update."""

    @patch.object(KalshiSource, "_build_resolution_df")
    @patch.object(KalshiSource, "_get_market")
    def test_new_id_appended(self, mock_market, mock_build, kalshi_source):
        """Tickers in dff not in dfq get appended with defaults."""
        mock_market.return_value = make_kalshi_api_market(ticker="new_001")
        mock_build.return_value = make_resolution_df(
            [{"id": "new_001", "date": "2024-06-01", "value": 0.5}]
        )
        dfq = make_question_df([{"id": "existing_001"}])
        dff = make_kalshi_fetch_df([{"id": "new_001"}])

        result = kalshi_source.update(dfq, dff)

        assert "new_001" in result.dfq["id"].values
        assert len(result.dfq) == 2
        new_row = result.dfq[result.dfq["id"] == "new_001"].iloc[0]
        assert new_row["freeze_datetime_value_explanation"] == "The market price."

    @patch.object(KalshiSource, "_build_resolution_df")
    @patch.object(KalshiSource, "_get_market")
    def test_existing_unresolved_updated(self, mock_market, mock_build, kalshi_source):
        """Unresolved question fields are updated from market details."""
        mock_market.return_value = make_kalshi_api_market(
            ticker="KXTEST-001",
            title="Updated question text",
            rules_primary="New rules",
        )
        mock_build.return_value = make_resolution_df(
            [{"id": "KXTEST-001", "date": "2024-06-01", "value": 0.65}]
        )
        dfq = make_question_df([{"id": "KXTEST-001", "resolved": False}])
        dff = make_kalshi_fetch_df([{"id": "KXTEST-001"}])

        result = kalshi_source.update(dfq, dff)

        row = result.dfq[result.dfq["id"] == "KXTEST-001"].iloc[0]
        assert row["question"] == "Updated question text"
        assert row["market_info_resolution_criteria"] == "New rules"
        assert "kalshi.com/markets/KXTEST-001" in row["url"]

    @patch.object(KalshiSource, "_build_resolution_df")
    @patch.object(KalshiSource, "_get_market")
    def test_market_becomes_resolved(self, mock_market, mock_build, kalshi_source):
        """Market with a terminal status marks the dfq row as resolved."""
        mock_market.return_value = make_kalshi_api_market(
            ticker="KXTEST-001",
            status="finalized",
            result="yes",
            settlement_ts="2026-01-13T05:00:00Z",
        )
        mock_build.return_value = make_resolution_df(
            [{"id": "KXTEST-001", "date": "2024-06-01", "value": 1.0}]
        )
        dfq = make_question_df([{"id": "KXTEST-001", "resolved": False}])
        dff = make_kalshi_fetch_df([{"id": "KXTEST-001"}])

        result = kalshi_source.update(dfq, dff)

        row = result.dfq[result.dfq["id"] == "KXTEST-001"].iloc[0]
        assert bool(row["resolved"]) is True
        assert "2026-01-13" in str(row["market_info_resolution_datetime"])

    @patch.object(KalshiSource, "_build_resolution_df")
    @patch.object(KalshiSource, "_get_market")
    def test_resolution_file_stored(self, mock_market, mock_build, kalshi_source):
        """Resolution file from _build_resolution_df is in result."""
        mock_market.return_value = make_kalshi_api_market(ticker="KXTEST-001")
        mock_build.return_value = make_resolution_df(
            [{"id": "KXTEST-001", "date": "2024-06-01", "value": 0.5}]
        )
        dfq = make_question_df([{"id": "KXTEST-001", "resolved": False}])
        dff = make_kalshi_fetch_df([{"id": "KXTEST-001"}])

        result = kalshi_source.update(dfq, dff)
        assert "KXTEST-001" in result.resolution_files

    @patch.object(KalshiSource, "_build_resolution_df")
    @patch.object(KalshiSource, "_get_market")
    def test_freeze_datetime_value_set(self, mock_market, mock_build, kalshi_source):
        """freeze_datetime_value is set to last value of resolution df."""
        mock_market.return_value = make_kalshi_api_market(ticker="KXTEST-001")
        mock_build.return_value = make_resolution_df(
            [
                {"id": "KXTEST-001", "date": "2024-06-01", "value": 0.3},
                {"id": "KXTEST-001", "date": "2024-06-02", "value": 0.75},
            ]
        )
        dfq = make_question_df([{"id": "KXTEST-001", "resolved": False}])
        dff = make_kalshi_fetch_df([{"id": "KXTEST-001"}])

        result = kalshi_source.update(dfq, dff)
        row = result.dfq[result.dfq["id"] == "KXTEST-001"].iloc[0]
        assert str(row["freeze_datetime_value"]) == "0.75"

    @patch.object(KalshiSource, "_build_resolution_df")
    @patch.object(KalshiSource, "_get_market")
    def test_build_resolution_returns_none(self, mock_market, mock_build, kalshi_source):
        """_build_resolution_df returning None: no resolution file stored."""
        mock_market.return_value = make_kalshi_api_market(ticker="KXTEST-001")
        mock_build.return_value = None
        dfq = make_question_df([{"id": "KXTEST-001", "resolved": False}])
        dff = make_kalshi_fetch_df([{"id": "KXTEST-001"}])

        result = kalshi_source.update(dfq, dff)
        assert "KXTEST-001" not in (result.resolution_files or {})

    @patch.object(KalshiSource, "_build_resolution_df")
    @patch.object(KalshiSource, "_get_market")
    def test_regenerates_missing_resolved_files(self, mock_market, mock_build, kalshi_source):
        """Resolved questions missing from storage get resolution files regenerated."""
        mock_market.return_value = make_kalshi_api_market(
            ticker="KXTEST-001", status="finalized", result="yes"
        )
        mock_build.return_value = make_resolution_df(
            [{"id": "KXTEST-001", "date": "2024-06-01", "value": 1.0}]
        )
        dfq = make_question_df(
            [
                {
                    "id": "KXTEST-001",
                    "resolved": True,
                    "market_info_resolution_datetime": "2024-07-01T00:00:00+00:00",
                }
            ]
        )
        dff = make_kalshi_fetch_df([{"id": "KXTEST-001"}])

        result = kalshi_source.update(dfq, dff, existing_resolution_ids=set())
        assert "KXTEST-001" in result.resolution_files

    @patch.object(KalshiSource, "_build_resolution_df")
    @patch.object(KalshiSource, "_get_market")
    def test_skips_resolved_already_in_storage(self, mock_market, mock_build, kalshi_source):
        """Resolved questions with files in storage are not re-fetched."""
        dfq = make_question_df(
            [
                {
                    "id": "KXTEST-001",
                    "resolved": True,
                    "market_info_resolution_datetime": "2024-07-01T00:00:00+00:00",
                }
            ]
        )
        dff = make_kalshi_fetch_df([{"id": "KXTEST-001"}])

        result = kalshi_source.update(dfq, dff, existing_resolution_ids={"KXTEST-001"})
        mock_market.assert_not_called()
        assert "KXTEST-001" not in (result.resolution_files or {})

    @patch.object(KalshiSource, "_build_resolution_df")
    @patch.object(KalshiSource, "_get_market")
    def test_caps_new_questions(self, mock_market, mock_build, kalshi_source):
        """New tickers exceeding the unresolved-pool cap are not all added."""
        mock_market.return_value = make_kalshi_api_market()
        mock_build.return_value = make_resolution_df(
            [{"id": "x", "date": "2024-06-01", "value": 0.5}]
        )
        # 1999 existing unresolved; only 1 new should fit under the 2000 cap.
        dfq = make_question_df([{"id": str(i)} for i in range(1999)])
        dff = make_kalshi_fetch_df([{"id": f"new_{i}"} for i in range(5)])

        result = kalshi_source.update(dfq, dff)
        assert len(result.dfq) == 2000

    @patch.object(KalshiSource, "_build_resolution_df")
    @patch.object(KalshiSource, "_get_market")
    def test_output_schema_valid(self, mock_market, mock_build, kalshi_source):
        """Output dfq passes QuestionFrame validation."""
        mock_market.return_value = make_kalshi_api_market(ticker="new_001")
        mock_build.return_value = make_resolution_df(
            [{"id": "new_001", "date": "2024-06-01", "value": 0.5}]
        )
        dfq = make_question_df([{"id": "existing_001"}])
        dff = make_kalshi_fetch_df([{"id": "new_001"}])

        result = kalshi_source.update(dfq, dff)
        QuestionFrame.validate(result.dfq)


# ---------------------------------------------------------------------------
# End-to-end resolution (constraint 4)
# ---------------------------------------------------------------------------


class TestKalshiEndToEnd:
    """Question set -> explode -> resolve_all -> dummy forecasts -> impute."""

    def test_resolution_passes(self, freeze_today):
        """A Kalshi question set resolves and dummy forecasts flow through imputation."""
        freeze_today(date(2025, 2, 1))

        # Build a question set: 2 standard + 1 combo Kalshi question, plus a data
        # question to seed the shared resolution date.
        question_set_df = make_question_set_df(
            [
                {"id": "m1", "source": "kalshi", "resolution_dates": "N/A"},
                {"id": "m2", "source": "kalshi", "resolution_dates": "N/A"},
                {"id": ("m1", "m2"), "source": "kalshi", "resolution_dates": "N/A"},
                {"id": "d1", "source": "fred", "resolution_dates": ["2025-01-08"]},
            ]
        )

        exploded = explode_question_set(question_set_df, "2025-01-01")
        exploded = exploded[exploded["source"] == "kalshi"].copy()
        assert len(exploded) > 0

        # Question bank: market resolves to yesterday's (Jan 31) value.
        dfq = make_question_df([{"id": "m1", "resolved": False}, {"id": "m2", "resolved": False}])
        dfr = make_resolution_df(
            [
                {"id": "m1", "date": "2025-01-01", "value": 0.3},
                {"id": "m1", "date": "2025-01-08", "value": 0.5},
                {"id": "m1", "date": "2025-01-31", "value": 0.7},
                {"id": "m2", "date": "2025-01-01", "value": 0.4},
                {"id": "m2", "date": "2025-01-08", "value": 0.6},
                {"id": "m2", "date": "2025-01-31", "value": 0.8},
            ]
        )
        question_bank = {"kalshi": SourceQuestionBank(dfq=dfq, dfr=dfr)}

        resolved, _ = resolve_all(
            exploded,
            question_bank=question_bank,
            sources={"kalshi": SOURCES["kalshi"]},
            forecast_due_date=date(2025, 1, 1),
        )
        assert len(resolved) > 0
        assert resolved["resolved_to"].notna().all()

        # Dummy forecasts: m1 provided, m2 missing (to exercise imputation).
        forecast_df = pd.DataFrame(
            {
                "id": ["m1", "m2"],
                "source": ["kalshi", "kalshi"],
                "direction": [(), ()],
                "forecast": [0.65, np.nan],
                "resolution_date": ["2025-01-08", "2025-01-08"],
            }
        )
        prepared = check_and_prepare_forecast_file(forecast_df, "2025-01-01", "test_org")
        merged = set_resolution_dates(prepared, resolved)
        result = impute_missing_forecasts(merged, "test_org", "test_model_org", "test_model")

        m1_rows = result[result["id"] == "m1"]
        assert len(m1_rows) > 0
        assert m1_rows.iloc[0]["forecast"] == 0.65
        assert bool(m1_rows.iloc[0]["imputed"]) is False

        m2_rows = result[result["id"] == "m2"]
        assert len(m2_rows) > 0
        assert m2_rows.iloc[0]["forecast"] == 0.5
        assert bool(m2_rows.iloc[0]["imputed"]) is True
