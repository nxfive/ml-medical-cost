MODEL_DEFAULT_CONFIG = {
    "cat_features": ["children", "region"],
    "num_features": ["age", "bmi"],
    "bin_features": ["sex", "smoker"],
    "models": {
        "LinearRegression": {
            "preprocess_num_features": True,
            "target_transformations": True,
            "params": {},
        },
        "KNeighborsRegressor": {
            "preprocess_num_features": True,
            "target_transformations": True,
            "params": {},
        },
        "DecisionTreeRegressor": {
            "preprocess_num_features": False,
            "target_transformations": False,
            "params": {},
        },
        "RandomForestRegressor": {
            "preprocess_num_features": False,
            "target_transformations": False,
            "params": {},
        },
    },
    "cv": {"n_splits": 5, "shuffle": True, "random_state": 42},
    "transformations": {
        "log": {"params": {}},
        "quantile": {"params": {}},
        "none": {"params": {}},
    },
}
