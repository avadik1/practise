from configs import make_configs


def make_base():
    return {"model": {"lr": 1e-3, "optimizer": "adam"}, "epochs": 10}


def test_lrs_set():
    cfgs = make_configs(make_base(), [1e-2, 1e-3, 1e-4])
    assert [c["model"]["lr"] for c in cfgs] == [1e-2, 1e-3, 1e-4]


def test_other_fields_kept():
    cfgs = make_configs(make_base(), [1e-2])
    assert cfgs[0]["epochs"] == 10
    assert cfgs[0]["model"]["optimizer"] == "adam"


def test_base_not_changed():
    base = make_base()
    make_configs(base, [1e-2, 1e-4])
    assert base == make_base()


def test_configs_independent():
    cfgs = make_configs(make_base(), [1e-2, 1e-3])
    cfgs[0]["model"]["optimizer"] = "sgd"
    assert cfgs[1]["model"]["optimizer"] == "adam"


def test_nested_dict_is_new_object():
    base = make_base()
    cfgs = make_configs(base, [1e-2])
    assert cfgs[0]["model"] is not base["model"]
