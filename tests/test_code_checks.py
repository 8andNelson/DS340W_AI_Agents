from src.orchestration.code_checks import find_leakage


def checks(script: str) -> list:
    return [f["check"] for f in find_leakage(script)]


def test_smote_before_the_split_is_critical_leakage():
    script = """
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)
"""
    findings = find_leakage(script)
    assert [f["check"] for f in findings] == ["resample_before_split"]
    assert findings[0]["severity"] == "critical"
    assert findings[0]["line"] == 2
    assert "fit_resample(X, y)" in findings[0]["evidence"]


def test_smote_on_training_data_after_the_split_is_fine():
    script = """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_res, y_res = SMOTE().fit_resample(X_train, y_train)
model.fit(X_res, y_res)
"""
    assert find_leakage(script) == []


def test_resampling_test_data_is_leakage():
    assert checks("X_test_res, y_test_res = SMOTE().fit_resample(X_test, y_test)\n") == ["resample_on_test"]


def test_resampling_train_and_test_combined_is_leakage():
    script = """
full = pd.concat([train, test])
Xr, yr = SMOTE().fit_resample(full.drop(columns="y"), full["y"])
"""
    assert checks(script) == ["resample_on_combined"]


def test_scaler_fitted_before_split_or_on_test_data():
    before = """
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""
    assert checks(before) == ["preprocessor_fit_before_split"]
    assert checks("X_test = scaler.fit_transform(X_test)\n") == ["preprocessor_fit_on_test"]
    assert checks("X_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n") == []


def test_test_set_used_for_early_stopping_or_training():
    assert checks("model.fit(X_train, y_train, eval_set=[(X_test, y_test)])\n") == ["test_used_for_early_stopping"]
    assert checks("model.fit(X_test, y_test)\n") == ["model_fit_on_test"]


def test_cross_validation_on_resampled_data():
    script = """
X_res, y_res = SMOTE().fit_resample(X_train, y_train)
scores = cross_val_score(model, X_res, y_res, cv=5)
"""
    assert checks(script) == ["cv_on_resampled_data"]


def test_unparseable_script_gives_no_findings():
    assert find_leakage("def broken(:\n") == []
    assert find_leakage("") == []
