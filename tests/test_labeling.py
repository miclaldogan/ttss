from ttss import TemporalWindowConfig, assign_temporal_label


def test_assign_temporal_label_boundaries() -> None:
    config = TemporalWindowConfig(pre_crime_frames=10, post_crime_frames=5)

    assert assign_temporal_label(90, 100, 120, config) == "pre-crime"
    assert assign_temporal_label(100, 100, 120, config) == "crime"
    assert assign_temporal_label(120, 100, 120, config) == "crime"
    assert assign_temporal_label(123, 100, 120, config) == "post-crime"
    assert assign_temporal_label(50, 100, 120, config) == "background"


def test_assign_temporal_label_validates_interval() -> None:
    try:
        assign_temporal_label(10, 20, 19)
    except ValueError as exc:
        assert "crime_end_frame" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid crime interval")
