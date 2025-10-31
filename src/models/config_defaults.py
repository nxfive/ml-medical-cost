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

OPTUNA_DEFAULT_CONFIG = {
    "optuna": {"trials": 10},
    "models": {
        "RandomForestRegressor": {
            "n_estimators": {"min": 50, "max": 100, "step": 10},
            "max_depth": {
                "min": 2,
                "max": 10,
            },
            "min_samples_split": {
                "min": 3,
                "max": 7,
            },
            "min_samples_leaf": {
                "min": 3,
                "max": 7,
            },
        },
        "DecisionTreeRegressor": {
            "max_depth": {
                "min": 2,
                "max": 10,
            },
            "min_samples_split": {
                "min": 3,
                "max": 7,
            },
            "min_samples_leaf": {"min": 3, "max": 7},
        },
        "KNeighborsRegressor": {
            "n_neighbors": {
                "min": 3,
                "max": 7,
            },
            "leaf_size": {
                "min": 10,
                "max": 20,
            },
        },
    },
}
