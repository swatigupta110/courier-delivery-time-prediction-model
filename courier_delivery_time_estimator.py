import os
import pandas as pd
from catboost import CatBoostRegressor
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="swatigupta110/courier-delivery-days-estimator-private",
    filename="model_courier_delivery_days_estimator.cb",
    token=os.getenv("HF_TOKEN")
)

model = CatBoostRegressor().load_model(model_path)

def process_and_predict(new_data):
  # model = CatBoostRegressor().load_model('catboost_model.cb')
  expected_columns = ['courier_partner', 'to_state', 'to_longitude', 'to_latitude', 'from_pincode', 'distance_in_km'] # Reorder columns to match the training data's feature order
  new_data = new_data[expected_columns].copy()                                                                        # Explicitly make a copy to avoid SettingWithCopyWarning
  delivery_days = model.predict(new_data)
  new_data['delivery_days'] = delivery_days
  return new_data

data = { "distance_in_km": [283.6075396208872],
        "from_pincode": ["282006"],
         "to_latitude": [26.849186],
         "to_longitude": [80.923539],
         "to_state": ["Uttar Pradesh"],
         "courier_partner": ["Xpressbees"]
}

new_data = pd.DataFrame(data)
result = process_and_predict(new_data)
result

# data = {
#     "distance_in_km": [
#         283.6075396208872,
#         370.42576077368744,
#         349.5648663686585,
#         624.9262115980044,
#         1021.1157441740133
#     ],
#     "from_pincode": [
#         282006,
#         284128,
#         301707,
#         301707,
#         301707
#     ],
#     "to_latitude": [
#         26.849186,
#         28.636221,
#         29.913626,
#         22.64863,
#         26.161637
#     ],
#     "to_longitude": [
#         80.923539,
#         77.292233,
#         73.865429,
#         75.294569,
#         86.892272
#     ],
#     "to_state": [
#         "Uttar Pradesh",
#         "Delhi",
#         "Rajasthan",
#         "Madhya Pradesh",
#         "Bihar"
#     ],
#     "courier_partner": [
#         "Xpressbees",
#         "Amazon Shipping",
#         "Amazon Shipping",
#         "DTDC",
#         "DTDC"
#     ]
# }

# new_data = pd.DataFrame(data)
# print(new_data)