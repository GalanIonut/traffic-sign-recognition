def test_tensorflow_imports_and_computes():
    import tensorflow as tf
    assert int(tf.reduce_sum(tf.constant([1, 2, 3]))) == 6



def test_config_constants():
    from traffic_signs import config
    assert config.NUM_CLASSES == 43
    assert config.SEED == 42
    assert config.DATA_DIR.name == "archive"



def test_fake_gtsrb_fixture(fake_gtsrb):
    assert (fake_gtsrb / "Train.csv").exists()
    assert len(list((fake_gtsrb / "Train").rglob("*.png"))) == 36
    assert len(list((fake_gtsrb / "Test").glob("*.png"))) == 6
