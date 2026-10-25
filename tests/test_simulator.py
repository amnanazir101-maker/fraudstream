import numpy as np

from fraudstream.simulator import DAY, FraudConfig, TransactionSimulator, make_world


def test_days_are_deterministic_and_independent():
    world = make_world(300, 800, seed=1)
    first = TransactionSimulator(world, seed=1)
    second = TransactionSimulator(world, seed=1)
    assert first.day(12).equals(second.day(12))
    assert first.day(12).equals(first.day(12))  # no hidden state between calls


def test_day_is_sorted_with_unique_ids():
    day = TransactionSimulator(make_world(300, 800, seed=2), seed=2).day(5)
    assert day["t"].is_monotonic_increasing
    assert day["tx_id"].is_unique
    assert ((day["t"] >= 5 * DAY) & (day["t"] < 6 * DAY)).all()


def test_classic_fraud_scenarios_appear():
    simulator = TransactionSimulator(make_world(800, 2000, seed=3), seed=3)
    history = [simulator.day(d) for d in range(30)]
    scenarios = set(np.concatenate([d["scenario"].unique() for d in history]))
    assert {0, 1, 2, 3} <= scenarios
    large = np.concatenate([d.loc[d["scenario"] == 1, "amount"] for d in history])
    assert (large > 220).all()


def test_card_testing_attack_starts_on_its_day():
    world = make_world(500, 1500, seed=4)
    simulator = TransactionSimulator(world, FraudConfig(attack_start_day=10), seed=4)
    assert (simulator.day(9)["scenario"] != 4).all()
    attack = simulator.day(10)
    testing = attack[attack["scenario"] == 4]
    assert len(testing) > 0
    assert (testing["amount"] <= 10).all()
    assert ((testing["t"] % DAY) < 6 * 3600).all()  # at night


def test_attack_does_not_change_earlier_days():
    world = make_world(300, 800, seed=5)
    calm = TransactionSimulator(world, seed=5)
    attacked = TransactionSimulator(world, FraudConfig(attack_start_day=20), seed=5)
    assert calm.day(19).equals(attacked.day(19))
