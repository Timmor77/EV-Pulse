"""Tests for feature engineering functions."""

import pandas as pd
import pytest

from src.features.build_features_v3 import CAT_FEATURES, add_context_features


class TestAddContextFeatures:
    """Test suite for the add_context_features function."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        dates = pd.date_range(start="2025-07-14 00:00", end="2025-07-14 23:45", freq="15min")
        df = pd.DataFrame(
            {
                "datetime": dates,
                "temperature": 25.0,
                "precipitation": 0.0,
                "solar_radiation": 500.0,
            }
        )
        return df

    @pytest.fixture
    def holiday_df(self):
        """Create a DataFrame with a US holiday."""
        # Christmas 2025
        dates = pd.date_range(start="2025-12-25 00:00", end="2025-12-25 23:45", freq="15min")
        df = pd.DataFrame(
            {
                "datetime": dates,
                "temperature": 15.0,
                "precipitation": 0.0,
                "solar_radiation": 400.0,
            }
        )
        return df

    @pytest.fixture
    def weekend_df(self):
        """Create a DataFrame for a Saturday."""
        # Saturday, July 19, 2025
        dates = pd.date_range(start="2025-07-19 00:00", end="2025-07-19 23:45", freq="15min")
        df = pd.DataFrame(
            {
                "datetime": dates,
                "temperature": 28.0,
                "precipitation": 0.0,
                "solar_radiation": 800.0,
            }
        )
        return df

    def test_basic_calendar_features(self, sample_df):
        """Test that basic calendar features are added."""
        result = add_context_features(sample_df)

        assert "hour" in result.columns
        assert "minute" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "year" in result.columns

        # Check first row values
        assert result.iloc[0]["hour"] == 0
        assert result.iloc[0]["minute"] == 0
        assert result.iloc[0]["month"] == 7
        assert result.iloc[0]["year"] == 2025

    def test_cyclical_features(self, sample_df):
        """Test that cyclical encodings are correct."""
        result = add_context_features(sample_df)

        # Check cyclical features exist
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "day_sin" in result.columns
        assert "day_cos" in result.columns
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns

        # Check values are in valid range [-1, 1]
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()
        assert result["day_sin"].between(-1, 1).all()
        assert result["day_cos"].between(-1, 1).all()

    def test_weekend_detection(self, weekend_df):
        """Test weekend detection."""
        result = add_context_features(weekend_df)

        assert "is_weekend" in result.columns
        assert result["is_weekend"].all()  # All rows should be weekend

    def test_weekday_detection(self, sample_df):
        """Test weekday detection (July 14, 2025 is Monday)."""
        result = add_context_features(sample_df)

        assert "is_weekend" in result.columns
        assert not result["is_weekend"].any()  # No rows should be weekend

    def test_holiday_detection(self, holiday_df):
        """Test US holiday detection."""
        result = add_context_features(holiday_df)

        assert "is_holiday" in result.columns
        assert result["is_holiday"].all()  # All rows should be holiday

    def test_business_time(self, sample_df):
        """Test business time calculation."""
        result = add_context_features(sample_df)

        assert "is_business_time" in result.columns
        assert "is_active_hour" in result.columns

        # At midnight (hour 0), should not be business time
        assert result.iloc[0]["is_business_time"] == 0
        assert result.iloc[0]["is_active_hour"] == 0

        # At 10am (hour 10), on a weekday, should be business time
        row_10am = result[result["hour"] == 10].iloc[0]
        assert row_10am["is_business_time"] == 1
        assert row_10am["is_active_hour"] == 1

    def test_business_time_on_weekend(self, weekend_df):
        """Test that weekends are not business time."""
        result = add_context_features(weekend_df)

        # Even during active hours, weekend should not be business time
        row_10am = result[result["hour"] == 10].iloc[0]
        assert row_10am["is_active_hour"] == 1
        assert row_10am["is_business_time"] == 0

    def test_interaction_features(self, sample_df):
        """Test interaction features."""
        result = add_context_features(sample_df)

        assert "hour_x_weekend" in result.columns
        assert "hour_x_month" in result.columns

        # On weekday, hour_x_weekend should be 0
        assert (result["hour_x_weekend"] == 0).all()

        # hour_x_month should be hour * month
        expected = result["hour"] * result["month"]
        assert (result["hour_x_month"] == expected).all()

    def test_cat_features_constant(self):
        """Test that CAT_FEATURES constant is properly defined."""
        expected_features = [
            "day_of_week",
            "month",
            "hour",
            "is_business_time",
            "is_holiday",
            "is_weekend",
            "is_active_hour",
        ]
        assert CAT_FEATURES == expected_features

    def test_output_row_count_preserved(self, sample_df):
        """Test that row count is preserved after transformation."""
        original_count = len(sample_df)
        result = add_context_features(sample_df)
        assert len(result) == original_count

    def test_handles_string_datetime(self):
        """Test that function handles string datetime column."""
        df = pd.DataFrame(
            {
                "datetime": ["2025-07-14 10:00:00", "2025-07-14 11:00:00"],
                "temperature": [25.0, 26.0],
            }
        )

        result = add_context_features(df)
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])

    def test_quarter_hour_encoding_is_distinct(self):
        """Cyclical time features must keep the four 15-minute slots distinct."""
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2025-07-14 10:00:00",
                        "2025-07-14 10:15:00",
                        "2025-07-14 10:30:00",
                        "2025-07-14 10:45:00",
                    ]
                )
            }
        )

        result = add_context_features(frame)

        assert result["minute"].tolist() == [0, 15, 30, 45]
        assert result["hour_sin"].nunique() == 4

    def test_utc_training_time_is_converted_to_pasadena_calendar_time(self):
        """Calendar features should describe the site, not the UTC clock."""
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2025-07-14 06:30:00", "2025-07-14 07:00:00"],
                    utc=True,
                )
            }
        )

        result = add_context_features(frame)

        assert result["hour"].tolist() == [23, 0]
        assert result["minute"].tolist() == [30, 0]
        assert result["day_of_week"].tolist() == [6, 0]
