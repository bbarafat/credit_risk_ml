from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

def build_preprocessor(
    continuous_features,
    categorical_features,
    pay_cols,
    education_cols,
    marital_cols,
):
   
    pay_order = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    education_order = [0, 1, 2, 3, 4, 5, 6]
    marital_order = [0, 1, 2, 3]

    pay_enc = OrdinalEncoder(
        categories=[pay_order] * len(pay_cols),
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    education_enc = OrdinalEncoder(
        categories=[education_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    marital_enc = OrdinalEncoder(
        categories=[marital_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_features),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_features),
            ("pay", pay_enc, pay_cols),
            ("edu", education_enc, education_cols),
            ("mar", marital_enc, marital_cols),
        ],
        remainder="drop",
    )